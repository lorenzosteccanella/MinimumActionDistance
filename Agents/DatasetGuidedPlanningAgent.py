import numpy as np
import torch
from typing import Optional, Tuple

from Agents.GoalAgent import GoalAgent
from Utils.Utils import infer_distance_device


class DatasetGuidedPlanningAgent(GoalAgent[np.ndarray, np.ndarray]):
    """
    CEM-based planning that samples action sequences from the offline dataset
    instead of random Gaussian noise. This produces realistic locomotion.
    """

    def __init__(
        self,
        distance,
        model,
        dataset: dict,  # Must have 'observations', 'actions'
        lookahead: int = 100,
        num_samples: int = 100,
        num_iterations: int = 10,
        num_elites: int = 10,
        step_penalty: float = 0.01,
        dist_batch_size: int = 512,
        cost_metric: str = "best_state",
        k_neighbors: int = 100,  # Number of nearest neighbors to consider
        similarity_fn=None,  # Optional fast distance fn(obs1, obs2) -> dists for NN search
    ):
        self.distance = distance
        self.similarity_fn = similarity_fn
        self.model = model
        self._device = infer_distance_device(distance)
        self.cost_metric = cost_metric
        self.lookahead = lookahead  # Set early - needed by _build_trajectory_indices

        # Store dataset
        self.dataset_obs = dataset["observations"]
        self.dataset_actions = dataset["actions"]
        self.dataset_terminals = dataset.get("terminals", np.zeros(len(self.dataset_obs)))

        # Build index for fast nearest neighbor search (just positions)
        self.dataset_positions = self.dataset_obs[:, :2].astype(np.float32)

        # Filter out flipped/unstable states
        self._compute_stability_mask()

        # Find trajectory boundaries (where terminals occur or large position jumps)
        self._build_trajectory_indices()

        # Model batch size
        if hasattr(self.model, "_num_envs"):
            self.model_batch_size = self.model._num_envs
        else:
            self.model_batch_size = 1

        # Adjust num_samples
        if num_samples % self.model_batch_size != 0:
            new_samples = max(
                self.model_batch_size,
                (num_samples // self.model_batch_size) * self.model_batch_size,
            )
            print(f"Adjusting num_samples from {num_samples} to {new_samples}")
            self.num_samples = new_samples
        else:
            self.num_samples = num_samples

        # Action space info
        if hasattr(self.model, "_real_env"):
            self.action_space = self.model._real_env.action_space
        else:
            self.action_space = self.model.vec_env.action_space

        self.action_dim = self.action_space.shape[-1]
        self.num_iterations = num_iterations
        self.num_elites = num_elites
        self.step_penalty = step_penalty
        self.dist_batch_size = dist_batch_size
        self.k_neighbors = k_neighbors

        self.plan = []
        self.done_sub_traj = False
        self.debug = False

        # Pre-compute embeddings for entire dataset (huge speedup for nearest neighbor)
        if self.similarity_fn is None:
            print("Pre-computing embeddings for dataset...")
            self._precompute_embeddings()
        else:
            print("Using custom similarity_fn for nearest-neighbor search (skipping embedding precomputation)")
            self.stable_embeddings = None

        print(
            f"DatasetGuidedPlanningAgent initialized: lookahead={lookahead}, "
            f"num_samples={self.num_samples}, k_neighbors={k_neighbors}, "
            f"dataset_size={len(self.dataset_obs)}, stable_states={self.stable_mask.sum()}"
        )

    def _quaternion_to_up_z(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Compute z-component of the "up" vector after quaternion rotation.
        Returns 1 for upright, -1 for flipped.
        quaternions shape: (N, 4) with (w, x, y, z) format
        """
        x, y = quaternions[:, 1], quaternions[:, 2]
        # up_z = 1 - 2*(x^2 + y^2) for rotating (0,0,1) by quaternion
        up_z = 1 - 2 * (x**2 + y**2)
        return up_z

    def _compute_stability_mask(self, up_z_threshold: float = 0.5, z_height_threshold: float = 0.25):
        """
        Compute a mask identifying stable (non-flipped) states in the dataset.

        A state is considered stable if:
        - The ant's "up" vector points upward (up_z > threshold)
        - The torso height is above minimum (z > threshold)
        """
        # Extract quaternion (w, x, y, z) from observations
        # Ant obs: [x, y, z, qw, qx, qy, qz, joints..., velocities...]
        quaternions = self.dataset_obs[:, 3:7]
        z_heights = self.dataset_obs[:, 2]

        # Compute stability metrics
        up_z = self._quaternion_to_up_z(quaternions)

        # Create stability mask
        upright_mask = up_z > up_z_threshold
        height_mask = z_heights > z_height_threshold
        self.stable_mask = upright_mask & height_mask

        # Store indices of stable states for efficient lookup
        self.stable_indices = np.where(self.stable_mask)[0]

        n_stable = self.stable_mask.sum()
        n_total = len(self.dataset_obs)
        print(f"Stability filter: {n_stable}/{n_total} states are stable ({100*n_stable/n_total:.1f}%)")
        print(f"  - Upright (up_z > {up_z_threshold}): {upright_mask.sum()}")
        print(f"  - Good height (z > {z_height_threshold}): {height_mask.sum()}")

    def _precompute_embeddings(self):
        """Pre-compute embeddings for stable dataset states only."""
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()

        # Only compute embeddings for stable states
        stable_obs = self.dataset_obs[self.stable_indices]
        all_embeddings = []

        for i in range(0, len(stable_obs), self.dist_batch_size):
            batch_obs = torch.from_numpy(
                stable_obs[i:i + self.dist_batch_size].astype(np.float32)
            ).to(self._device)
            # eval_embed_state handles eval mode and no_grad internally
            batch_embed = self.distance.eval_embed_state(batch_obs)
            all_embeddings.append(batch_embed.cpu())

        # Store as single tensor on GPU for fast distance computation
        self.stable_embeddings = torch.cat(all_embeddings, dim=0).to(self._device)
        print(f"Pre-computed {len(self.stable_embeddings)} stable embeddings, shape: {self.stable_embeddings.shape}")

    def _build_trajectory_indices(self):
        """Identify trajectory start indices in the dataset."""
        # Trajectories end at terminals or large position discontinuities
        terminals = self.dataset_terminals.astype(bool)
        pos_diff = np.linalg.norm(np.diff(self.dataset_positions, axis=0), axis=1)
        jumps = pos_diff > 2.0  # Position jump > 2m indicates new trajectory
        
        # Trajectory boundaries
        boundaries = np.zeros(len(self.dataset_obs), dtype=bool)
        boundaries[:-1] = terminals[:-1] | jumps
        boundaries[-1] = True
        
        # Store valid ranges (start, end) for each trajectory
        self.traj_ranges = []
        start = 0
        for i, is_boundary in enumerate(boundaries):
            if is_boundary:
                if i - start >= self.lookahead:  # Only keep long enough trajectories
                    self.traj_ranges.append((start, i))
                start = i + 1
        
        print(f"Found {len(self.traj_ranges)} valid trajectories in dataset")

    def _find_nearest_indices(self, position: np.ndarray, k: int) -> np.ndarray:
        """Find k nearest dataset states to the given position."""
        dists = np.linalg.norm(self.dataset_positions - position[:2], axis=1)
        return np.argsort(dists)[:k]
    
    def _find_learned_nearest_indices(self, state: np.ndarray, k: int) -> np.ndarray:
        """Find k nearest STABLE dataset states using pre-computed embeddings."""
        # Use batched version with single state
        return self._find_learned_nearest_indices_batched(state[np.newaxis, :], k)[0]

    def _find_learned_nearest_indices_batched(self, states: np.ndarray, k: int) -> np.ndarray:
        """
        Find k nearest STABLE dataset states for multiple query states.

        Args:
            states: (num_queries, state_dim) array of query states
            k: number of neighbors per query

        Returns:
            (num_queries, k) array of dataset indices
        """
        if self.similarity_fn is not None:
            return self._find_nearest_with_dist_fn(states, k)

        num_queries = states.shape[0]
        states_tensor = torch.from_numpy(states.astype(np.float32)).to(self._device)

        # Embed all query states at once
        states_embed = self.distance.eval_embed_state(states_tensor)  # (num_queries, embed_dim)

        # Compute distances: for each query, compute dist to all stable embeddings
        # Result shape: (num_queries, num_stable)
        all_dists = []
        for i in range(0, len(self.stable_embeddings), self.dist_batch_size):
            batch_embed = self.stable_embeddings[i:i + self.dist_batch_size]  # (batch, embed_dim)
            batch_size = batch_embed.shape[0]

            # Expand for broadcasting: states (Q, 1, D) vs batch (1, B, D) -> (Q, B)
            batch_dists = []
            for q in range(num_queries):
                q_embed = states_embed[q:q+1].expand(batch_size, -1)  # (batch, embed_dim)
                dists = self.distance.eval_z_dist(q_embed, batch_embed)  # (batch,)
                batch_dists.append(dists)
            all_dists.append(torch.stack(batch_dists, dim=0))  # (num_queries, batch)

        # Concatenate on GPU: (num_queries, num_stable)
        dists = torch.cat(all_dists, dim=1)

        # Get top-k for each query
        _, topk_indices = torch.topk(dists, k, largest=False, dim=1)  # (num_queries, k)
        stable_nearest = topk_indices.cpu().numpy()

        # Map back to original dataset indices
        return self.stable_indices[stable_nearest]

    def _find_nearest_with_dist_fn(self, states: np.ndarray, k: int) -> np.ndarray:
        """Find k nearest stable dataset states using the custom similarity_fn."""
        stable_obs = self.dataset_obs[self.stable_indices]
        num_queries = states.shape[0]
        all_nearest = []
        for q in range(num_queries):
            dists = self.similarity_fn(states[q], stable_obs)  # (num_stable,)
            topk_idx = np.argpartition(dists, k)[:k]
            topk_idx = topk_idx[np.argsort(dists[topk_idx])]
            all_nearest.append(self.stable_indices[topk_idx])
        return np.stack(all_nearest, axis=0)

    def _sample_action_sequences_from_dataset(
        self, state: np.ndarray, num_sequences: int
    ) -> np.ndarray:
        """
        Sample action sequences from the dataset starting from states similar to current state.
        """
        near_indices = self._find_learned_nearest_indices(state, self.k_neighbors)
        return self._sample_sequences_from_neighbors(near_indices, num_sequences)

    def _sample_action_sequences_from_states_batched(
        self, states: np.ndarray, num_sequences_per_state: int
    ) -> np.ndarray:
        """
        Sample action sequences for multiple query states in one batched call.

        Args:
            states: (num_states, state_dim) array of query states
            num_sequences_per_state: sequences to sample per state

        Returns:
            (num_states * num_sequences_per_state, lookahead, action_dim) array
        """
        # Batch nearest neighbor search for all states at once
        all_near_indices = self._find_learned_nearest_indices_batched(states, self.k_neighbors)

        all_sequences = []
        for near_indices in all_near_indices:
            seqs = self._sample_sequences_from_neighbors(near_indices, num_sequences_per_state)
            all_sequences.append(seqs)

        return np.concatenate(all_sequences, axis=0)

    def _sample_sequences_from_neighbors(
        self, near_indices: np.ndarray, num_sequences: int
    ) -> np.ndarray:
        """Sample action sequences given pre-computed neighbor indices."""
        action_sequences = []
        for _ in range(num_sequences):
            # Pick a random nearby starting point
            start_idx = np.random.choice(near_indices)

            # Find which trajectory this belongs to and get valid range
            valid_end = start_idx + self.lookahead
            for traj_start, traj_end in self.traj_ranges:
                if traj_start <= start_idx < traj_end:
                    valid_end = min(valid_end, traj_end)
                    break

            # Extract actions, padding if necessary
            actions = []
            for t in range(self.lookahead):
                idx = start_idx + t
                if idx < valid_end and idx < len(self.dataset_actions):
                    actions.append(self.dataset_actions[idx])
                else:
                    # Pad with last valid action or zeros
                    actions.append(actions[-1] if actions else np.zeros(self.action_dim))

            action_sequences.append(np.stack(actions, axis=0))

        return np.stack(action_sequences, axis=0)

    def _plan_with_cem(self, state: np.ndarray, goal_tensor: torch.Tensor) -> tuple:
        print("Planning with dataset-guided CEM...")

        best_traj = None
        best_score = float("inf")
        predicted_final_state = None

        with torch.no_grad():
            init_state_t = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
            init_dist = self.distance.eval_dist(init_state_t, goal_tensor.to(self._device)).item()
        print(f"Initial distance: {init_dist:.4f}, pos: {state[:2]}, goal: {goal_tensor[0, :2].cpu().numpy()}")

        # Initialize with dataset samples
        action_sequences = self._sample_action_sequences_from_dataset(state, self.num_samples)

        for iter_idx in range(self.num_iterations):
            traj_states_batches = []

            for start_idx in range(0, self.num_samples, self.model_batch_size):
                end_idx = start_idx + self.model_batch_size
                batch_actions = action_sequences[start_idx:end_idx]

                current_batch_states = np.tile(state, (self.model_batch_size, 1))
                has_terminated = np.zeros(self.model_batch_size, dtype=bool)
                batch_traj_list = []

                # First step: use predict_batch to set initial simulator state
                next_obs, _, dones = self.model.predict_batch(
                    current_batch_states, batch_actions[:, 0, :]
                )
                was_alive = ~has_terminated
                should_update = was_alive & ~dones.astype(bool)
                if np.any(should_update):
                    current_batch_states[should_update] = next_obs[should_update]
                has_terminated |= dones.astype(bool)
                batch_traj_list.append(current_batch_states.copy())

                # Subsequent steps: use step_batch (no state reset overhead)
                use_step_batch = hasattr(self.model, "step_batch")
                for t in range(1, self.lookahead):
                    if use_step_batch:
                        next_obs, _, dones = self.model.step_batch(batch_actions[:, t, :])
                    else:
                        next_obs, _, dones = self.model.predict_batch(
                            current_batch_states, batch_actions[:, t, :]
                        )

                    was_alive = ~has_terminated
                    should_update = was_alive & ~dones.astype(bool)
                    if np.any(should_update):
                        current_batch_states[should_update] = next_obs[should_update]

                    has_terminated |= dones.astype(bool)
                    batch_traj_list.append(current_batch_states.copy())

                traj_states_batches.append(np.stack(batch_traj_list, axis=0))

            traj_states_np = np.concatenate(traj_states_batches, axis=1)

            # Evaluate costs (keep on GPU until final reshape)
            flat_states = traj_states_np.reshape(-1, traj_states_np.shape[-1])
            goal_dev = goal_tensor.to(self._device)
            dists_list = []
            for i in range(0, flat_states.shape[0], self.dist_batch_size):
                b_states = torch.from_numpy(flat_states[i:i+self.dist_batch_size]).float().to(self._device)
                g_batch = goal_dev.expand(b_states.shape[0], -1)
                with torch.no_grad():
                    dists_list.append(self.distance.eval_dist(b_states, g_batch))

            all_dists = torch.cat(dists_list).cpu().numpy().reshape(self.lookahead, self.num_samples)
            penalty = self.step_penalty * np.arange(self.lookahead)[:, None]
            dists_with_penalty = all_dists + penalty
            min_indices = np.argmin(dists_with_penalty, axis=0)
            costs = dists_with_penalty[min_indices, np.arange(self.num_samples)]

            # Select elites
            elite_ids = costs.argsort()[:self.num_elites]

            if costs[elite_ids[0]] < best_score:
                best_score = costs[elite_ids[0]]
                best_idx = elite_ids[0]
                best_t = min_indices[best_idx]
                best_traj = action_sequences[best_idx, :best_t + 1, :]
                predicted_final_state = traj_states_np[best_t, best_idx]

            if iter_idx % 5 == 0 or iter_idx == self.num_iterations - 1:
                best_pos = traj_states_np[min_indices[elite_ids[0]], elite_ids[0], :2]
                print(f"  Iter {iter_idx+1}: cost={costs[elite_ids[0]]:.2f}, pos={best_pos}, steps={min_indices[elite_ids[0]]+1}")

            # Resample from elite endpoints (batched for efficiency)
            elite_endpoints = traj_states_np[-1, elite_ids, :]
            samples_per_elite = self.num_samples // self.num_elites
            new_sequences = self._sample_action_sequences_from_states_batched(
                elite_endpoints, samples_per_elite
            )[:self.num_samples]
            
            # Keep some elites
            num_keep = min(self.num_elites, self.num_samples // 4)
            action_sequences = np.concatenate(
                [action_sequences[elite_ids[:num_keep]], new_sequences[:self.num_samples - num_keep]],
                axis=0,
            )

        return best_traj.tolist(), predicted_final_state

    def act(self, state: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
            (action, predicted_state) where predicted_state is the agent's
            predicted final position of the current sub-trajectory.
            Check self.done_sub_traj after each call to know if the
            sub-trajectory just finished.
        """
        goal_tensor = torch.from_numpy(goal).float().unsqueeze(0).to(self._device)

        if len(self.plan) == 0:
            self.plan, self.predicted_final_state = self._plan_with_cem(state, goal_tensor)

        action = np.array(self.plan.pop(0))
        self.done_sub_traj = len(self.plan) == 0
        return action, self.predicted_final_state

    def reset(self) -> None:
        self.plan = []
        self.done_sub_traj = False
