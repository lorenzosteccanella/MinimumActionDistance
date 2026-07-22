import numpy as np
import torch
from collections import deque
from tqdm import trange
from typing import Union, Tuple, List, Optional

class ErDist:
    """
    An efficient, vectorized experience replay buffer for storing trajectories and sampling batches.
    This buffer is optimized for training models that measure distances between states.
    """

    def __init__(self,
                 max_n_trajectories: int = 10000,
                 trajectories_list: Optional[List] = None,
                 prioritization: bool = False,
                 episode_len: Optional[int] = None):
        """
        Initializes the experience replay buffer.

        :param max_n_trajectories: The maximum number of trajectories to store.
        :param trajectories_list: Optional list of trajectories to pre-populate the buffer.
        :param prioritization: Whether to enable Prioritized Experience Replay (PER).
        :param episode_len: The maximum length of an episode (used for PER buffer sizing).
                             Required if prioritization is enabled.
        """
        self.prioritization = prioritization
        # The deque now stores tuples of (states_tensor, next_states_tensor, length)
        self.trajectories = deque(maxlen=max_n_trajectories)
        self.feature_dim = None

        if self.prioritization:
            assert episode_len is not None, "episode_len must be provided when using prioritization."
            try:
                from torchrl.data import ReplayBuffer, ListStorage
                from torchrl.data.replay_buffers.samplers import PrioritizedSampler
            except ImportError:
                raise ImportError("torchrl is not installed. Please install it to use prioritization: pip install torchrl")

            max_size_er = max_n_trajectories * episode_len * 4
            self.per = ReplayBuffer(
                storage=ListStorage(max_size=max_size_er),
                sampler=PrioritizedSampler(max_capacity=max_size_er, alpha=0.8, beta=0.4),
                collate_fn=lambda x: x,
            )

        if trajectories_list is not None:
            self.add_trajectories(trajectories_list)

    def add_trajectories(self, trajectories: List):
        """
        Adds a list of trajectories to the buffer.

        :param trajectories: The list of trajectories to add.
        """
        for traj in trange(len(trajectories), desc='Adding trajectories to ER', leave=True):
            self.add_trajectory(trajectory=trajectories[traj])

    def add_trajectory(self, trajectory: Union[List, Tuple], max_d_c: Optional[int] = None):
        """
        Adds a single trajectory to the buffer, converting it to an efficient tensor format.

        :param trajectory: The trajectory to add.
        :param max_d_c: The maximum distance constraint for generating prioritized transitions.
                        Required if prioritization is enabled.
        """
        if not trajectory:
            return

        # --- Convert list of dicts to tensors for efficient storage and access ---
        # This is a one-time cost during insertion that pays off massively during sampling.
        states = torch.stack([step["s"] for step in trajectory])
        next_states = torch.stack([step["s_"] for step in trajectory])
        
        if self.feature_dim is None:
            self.feature_dim = states.shape[1]

        traj_len = len(trajectory)
        self.trajectories.append((states, next_states, traj_len))

        if self.prioritization:
            assert max_d_c is not None, "max_d_c must be provided when using prioritization."
            
            # --- Vectorized creation of transitions for PER ---
            # Create all possible start and distance combinations
            i = torch.arange(traj_len)
            j = torch.arange(1, max_d_c + 1)
            start_indices, dists = torch.meshgrid(i, j, indexing='ij')

            # Calculate end indices and filter out-of-bounds pairs
            end_indices = start_indices + dists
            valid_mask = end_indices < traj_len

            start_indices = start_indices[valid_mask]
            end_indices = end_indices[valid_mask]
            dists = dists[valid_mask]
            
            # Add all valid transitions to PER buffer in a batch
            if len(start_indices) > 0:
                transitions = {
                    "s": states[start_indices],
                    "s_": states[end_indices], # s_ is the state at the end of the distance
                    "d": dists.float()
                }
                self.per.add(transitions)

    def _get_valid_trajectories(self, batch_size: int) -> np.ndarray:
        """Helper to get random indices of stored trajectories."""
        assert len(self.trajectories) > 0, "Cannot sample from an empty buffer."
        assert batch_size > 0, "Batch size must be greater than 0."
        return np.random.randint(len(self.trajectories), size=batch_size)

    def _sample_indices(self, batch_size: int, d_thresh: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized sampling of trajectory and step indices, guaranteed to have d >= 1."""
        traj_indices = self._get_valid_trajectories(batch_size)
        
        lengths = np.array([self.trajectories[i][2] for i in traj_indices])
        
        # Sample start points, ensuring s1 is not the last state.
        # This is a critical first step to guarantee d >= 1.
        s1_indices = np.random.randint(lengths - 1)

        if d_thresh is None:
            # Sample end points anywhere *after* s1.
            s2_indices = np.random.randint(s1_indices + 1, lengths)
        else:
            # Sample end points within the distance threshold.
            max_range = np.minimum(s1_indices + d_thresh + 1, lengths)
            
            # Calculate the size of the valid offset range [1, max_offset].
            offset_range_size = max_range - s1_indices
            
            # We clip to ensure the lower bound of randint is not >= the upper bound.
            min_offset = 1
            # +1 because randint's high is exclusive
            max_offset = np.clip(offset_range_size, a_min=min_offset + 1, a_max=None) 
            offset = np.random.randint(low=min_offset, high=max_offset)
            
            s2_indices = s1_indices + offset
        
        return traj_indices, s1_indices, s2_indices

    def get_batch(self, batch_size: int, d_thresh: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch of (s1, s2, distance) tuples.

        :param batch_size: The batch size.
        :param d_thresh: An optional distance threshold for sampling s2.
        :return: A tuple of tensors (s1_batch, s2_batch, distance_batch).
        """
        traj_indices, s1_indices, s2_indices = self._sample_indices(batch_size, d_thresh)

        # Gather data using list comprehensions (much faster than loops)
        s1_list = [self.trajectories[t_idx][0][s1_idx] for t_idx, s1_idx in zip(traj_indices, s1_indices)]
        s2_list = [self.trajectories[t_idx][0][s2_idx] for t_idx, s2_idx in zip(traj_indices, s2_indices)]
        
        # Create tensors in one bulk operation
        s1_b = torch.stack(s1_list)
        s2_b = torch.stack(s2_list)
        d_traj_b = torch.from_numpy(s2_indices - s1_indices).float()

        return s1_b, s2_b, d_traj_b

    def get_triplet_batch(self, batch_size: int, d_thresh: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch of (s, s_next, g, distance) tuples for triplet loss.

        :param batch_size: The batch size.
        :param d_thresh: An optional distance threshold for sampling the goal g.
        :return: A tuple of tensors (s_batch, s_next_batch, goal_batch, distance_batch).
        """
        traj_indices, s1_indices, s2_indices = self._sample_indices(batch_size, d_thresh)
        
        s_list = [self.trajectories[t_idx][0][s1_idx] for t_idx, s1_idx in zip(traj_indices, s1_indices)]
        s_next_list = [self.trajectories[t_idx][1][s1_idx] for t_idx, s1_idx in zip(traj_indices, s1_indices)]
        g_list = [self.trajectories[t_idx][0][s2_idx] for t_idx, s2_idx in zip(traj_indices, s2_indices)]
        
        s_b = torch.stack(s_list)
        s__b = torch.stack(s_next_list)
        g_b = torch.stack(g_list)
        d_traj_b = torch.from_numpy(s2_indices - s1_indices).float()

        return s_b, s__b, g_b, d_traj_b
        
    def get_state_batch(self, batch_size: int) -> torch.Tensor:
        """
        Samples a single random batch of states from the buffer.

        :param batch_size: The batch size.
        :return: A tensor of states.
        """
        traj_indices = self._get_valid_trajectories(batch_size)
        lengths = np.array([self.trajectories[i][2] for i in traj_indices])
        state_indices = np.random.randint(lengths)

        s_list = [self.trajectories[t_idx][0][s_idx] for t_idx, s_idx in zip(traj_indices, state_indices)]
        
        return torch.stack(s_list)
        
    def get_states_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples two independent random batches of states from the buffer.

        :param batch_size: The batch size.
        :return: A tuple containing two tensors of states.
        """
        s_b_1 = self.get_state_batch(batch_size)
        s_b_2 = self.get_state_batch(batch_size)
        return s_b_1, s_b_2

    def get_prioritized_batch_c(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch from the Prioritized Experience Replay buffer.

        :param batch_size: The batch size.
        :return: A tuple of (s1, s2, distance, indices).
        """
        assert self.prioritization, "Prioritized replay is not enabled."
        assert len(self.per) >= batch_size, f"Not enough samples in PER buffer to get a batch of size {batch_size}. Have {len(self.per)}."

        transitions, info = self.per.sample(batch_size, return_info=True)
        
        # Efficiently stack the batched data from torchrl
        s1_c = torch.stack(transitions["s"])
        s2_c = torch.stack(transitions["s_"])
        d_c = transitions["d"]

        return s1_c, s2_c, d_c, info["index"]

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """
        Updates the priorities of transitions in the PER buffer.

        :param indices: The indices of the transitions to update.
        :param priorities: The new priority values.
        """
        assert self.prioritization, "Prioritized replay is not enabled."
        self.per.update_priority(indices, priorities)

    def __len__(self):
        """Returns the number of trajectories currently in the buffer."""
        return len(self.trajectories)