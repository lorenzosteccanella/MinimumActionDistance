# This file is based on the original implementation of the Hilbert representation model
# produced by Park et al. (2024). More specifically, it is based directly on the implementation
# details specified in the paper and code published by the authors.
#
# The following code file was particularly useful in helping us develop an accurate and
# faithful procedure for training Hilbert representation models:
# - https://github.com/seohongpark/HILP/blob/master/hilp_gcrl/src/agents/hilp.py

import copy
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Callable
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage

from Utils.Utils import vector_norm
from Distances.Distance import Distance
from DistExpReplay.ErDist import ErDist
from Models.HilbertModels import HilbertEmbeddingModel
from ObjectiveFunctions.ExpectileLoss import ExpectileLoss

PROP_FUTURE = 0.625
PROP_RANDOM = 1 - PROP_FUTURE


class HilbertDistance(Distance):
    def __init__(self, config: dict, network_1: nn.Module, network_2: nn.Module, norm: Callable = None):
        # Hyperparameters and PyTorch config.
        self.batch_size = config["batch_size"]
        self.device = config["device"]

        # HER parameterts.
        self.her_num_goals = config["her_num_goals"]

        # IQL parameters.
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.expectile = config["expectile"]

        # Network 1.
        self.online_1 = copy.deepcopy(network_1).to(self.device)
        self.target_1 = copy.deepcopy(network_1).to(self.device)
        self.target_1.load_state_dict(self.online_1.state_dict())
        for p in self.target_1.parameters():
            p.requires_grad = False

        # Network 2.
        self.online_2 = copy.deepcopy(network_2).to(self.device)
        self.target_2 = copy.deepcopy(network_2).to(self.device)
        self.target_2.load_state_dict(self.online_2.state_dict())
        for p in self.target_2.parameters():
            p.requires_grad = False

        # Online network optimiser.
        self.optimiser = torch.optim.RAdam(
            list(self.online_1.parameters()) + list(self.online_2.parameters()), lr=self.alpha
        )

        # Empty list of raw trajectories.
        self.trajectories = []

        self.exp_rep = ErDist(
            max_n_trajectories=config["er_max_n_traj"],
            prioritization=config.get("prioritization", False),
        )

        # Buffer for storing processed HER experience.
        self.her_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=config["buffer_size"],
                device=config["device"],
            )
        )

        # Save config for later use.
        self.config = config

        if norm is not None:
            self.norm = norm
        else:
            self.norm = vector_norm

    def add_trajectories(self, trajectories: list):
        for trajectory in trajectories:
            self.add_trajectory(trajectory)

    def add_trajectory(self, trajectory: list):
        self.trajectories.append(trajectory)
        self.exp_rep.add_trajectory(trajectory, max_d_c=self.config.get("max_dist_con"))

    def train(
        self,
        steps: int = 1,
        process_her_trajectories: bool = False,
        clear_trajectories: bool = False,
        clear_her_buffer: bool = False,
        verbose=False,
    ):
        # Create HER replay buffer based on current trajectories.
        if process_her_trajectories:
            if verbose:
                print("\nRelabelling trajectories using HER...")
            self._her_process_trajectories(verbose)

        # Train Hilbert embedding using HER replay buffer.
        losses = [0] * steps
        if steps > 0:
            if verbose:
                print("\nTraining Hilbert representation...")
            pbar = tqdm(total=steps, desc="Hilbert Model | Loss = 0.0", unit="update", disable=not verbose)
            for i in range(steps):
                loss = self._train_hilbert_model()
                losses[i] = loss
                pbar.update(1)
                pbar.set_description(f"Hilbert Model | Loss = {loss:.4f}")
            pbar.close()

            if verbose:
                print("\nFinished training Hilbert representation!")

        # Clear trajectory list.
        if clear_trajectories:
            if verbose:
                print("\nClearing trajectories...")
            self.trajectories.clear()

        # Clear and HER buffer.
        if clear_her_buffer:
            if verbose:
                print("\nClearing HER buffer...")
            self.her_buffer = ReplayBuffer(
                storage=LazyTensorStorage(
                    max_size=self.config["buffer_size"],
                    device=self.config["device"],
                )
            )

        return losses

    def _her_process_trajectories(self, verbose: bool = False):
        # Create a tensor of states, actions, and next-states for each trajectory.
        traj_states = [torch.stack([experience["s"] for experience in trajectory]) for trajectory in self.trajectories]
        traj_next_states = [
            torch.stack([experience["s_"] for experience in trajectory]) for trajectory in self.trajectories
        ]

        # Create a combined tensor containing all states across all trajectories.
        all_states = torch.cat(traj_states, dim=0)

        # Process each trajectory.
        for i in tqdm(range(len(self.trajectories)), desc="HER Processing", unit="trajectory", disable=not verbose):
            traj_len = len(self.trajectories[i])

            # Create tensors that duplicates each time-step her_num_goals + 1 times.
            her_states = traj_states[i].repeat(self.her_num_goals + 1, 1)
            her_next_states = traj_next_states[i].repeat(self.her_num_goals + 1, 1)

            # Create a tensor to store the goals for each time-step.
            her_goals = torch.zeros((self.her_num_goals + 1) * traj_len, self.config["obs_size"])

            # The first goal for each time-step is the final state of the trajectory.
            her_goals[0:traj_len] = traj_states[i][-1]

            # The next PROP_FUTURE * her_num_goals goals are sampled from future states in the same trajectory.
            # TODO: Consider how the last few states in a trajectory will have fewer future states to sample from.
            time_steps = torch.arange(0, traj_len).repeat(int(PROP_FUTURE * self.her_num_goals))
            future_offsets = torch.from_numpy(
                np.random.geometric(1 - self.gamma, (traj_len * int(PROP_FUTURE * self.her_num_goals)))
            )
            sampled_future_time_steps = torch.minimum(
                time_steps + future_offsets, torch.full_like(time_steps, traj_len - 1)
            )

            # Extract future goals based on sampled indices.
            sampled_future_goals = traj_states[i][sampled_future_time_steps.flatten()].view(
                traj_len, int(PROP_FUTURE * self.her_num_goals), -1
            )
            sampled_future_goals = sampled_future_goals.view(-1, self.config["obs_size"])

            # The remaining goals are sampled randomly from the entire dataset.
            remaining_goals_count = self.her_num_goals - sampled_future_goals.size(0) // traj_len
            random_indices = torch.randint(0, len(all_states), (remaining_goals_count * traj_len,))

            # Reshape `random_indices` to match the per-step structure.
            random_goals = all_states[random_indices].view(traj_len, remaining_goals_count, -1)

            # Flatten `random_goals` for consistent appending to `her_goals`.
            random_goals = random_goals.view(-1, self.config["obs_size"])

            # Fill in `her_goals` with the sampled future and random goals.
            her_goals[traj_len : traj_len + sampled_future_goals.size(0)] = sampled_future_goals
            her_goals[traj_len + sampled_future_goals.size(0) :] = random_goals

            # Compute terminality for each goal based on whether the state is the goal.
            her_terminals = (her_states == her_goals).all(dim=-1)

            # Set the reward to -1.0 for each action taken.
            her_rewards = (-torch.ones((self.her_num_goals + 1) * traj_len) * (1 - her_terminals.float())).unsqueeze(-1)

            # Add the re-labelled experiences to the HER buffer.
            data = TensorDict(
                {
                    "s": her_states,
                    "r": her_rewards,
                    "s_": her_next_states,
                    "done": her_terminals,
                    "goal": her_goals,
                },
                batch_size=len(her_states),
            ).to(self.device)
            self.her_buffer.extend(data)

    def _train_hilbert_model(self):
        loss = self._update_online_network()
        self._update_target_network()

        return loss

    def _update_online_network(self):
        # Hilbert representation update step.
        # Based on JAX code from: https://github.com/seohongpark/HILP/blob/master/hilp_gcrl/src/agents/hilp.py

        batch = self.her_buffer.sample(self.batch_size)
        states = batch["s"]
        rewards = batch["r"]
        next_states = batch["s_"]
        terminals = batch["done"]
        goals = batch["goal"]

        loss = torch.tensor([0])

        next_v1, next_v2 = self._value_target(next_states, goals)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.gamma * next_v.squeeze() * torch.logical_not(terminals.flatten())

        v1_t, v2_t = self._value_target(states, goals)
        v_t = (v1_t + v2_t) / 2.0
        advantage = q - v_t

        q1 = rewards + self.gamma * next_v1 * torch.logical_not(terminals.flatten())
        q2 = rewards + self.gamma * next_v2 * torch.logical_not(terminals.flatten())
        v1, v2 = self._value(states, goals)

        loss_function = ExpectileLoss(self.expectile)
        value_loss_1 = loss_function(advantage, q1 - v1)
        value_loss_2 = loss_function(advantage, q2 - v2)
        loss = (value_loss_1 + value_loss_2) / 2.0

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()

    @torch.no_grad()
    def _update_target_network(self):
        tau = self.tau

        # Network 1.
        for p_o, p_t in zip(self.online_1.parameters(), self.target_1.parameters()):
            p_t.data.copy_(tau * p_o.data + (1 - tau) * p_t.data)
            p_t.requires_grad = False

        # Network 2.
        for p_o, p_t in zip(self.online_2.parameters(), self.target_2.parameters()):
            p_t.data.copy_(tau * p_o.data + (1 - tau) * p_t.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def eval_dist(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        s1, s2 = s1.to(self.device), s2.to(self.device)  # Ensure inputs are on the same device as the model.
        return self.norm(self.eval_embed_state(s1), self.eval_embed_state(s2))

    @torch.no_grad()
    def eval_z_dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.norm(z1, z2)

    @torch.no_grad()
    def eval_embed_state(self, s: torch.Tensor) -> torch.Tensor:
        self.online_1.eval()
        self.online_2.eval()
        s = s.to(self.device)
        return (self.online_1(s) + self.online_2(s)) / 2.0

    def _value(self, s1, s2):
        s1, s2 = s1.to(self.device), s2.to(self.device)
        return (-self.norm(self.online_1(s1), self.online_1(s2)), -self.norm(self.online_2(s1), self.online_2(s2)))

    @torch.no_grad()
    def _value_target(self, s1, s2):
        s1, s2 = s1.to(self.device), s2.to(self.device)
        return (-self.norm(self.target_1(s1), self.target_1(s2)), -self.norm(self.target_2(s1), self.target_2(s2)))

    def save(self, save_path: str) -> None:
        """
        Save the Hilbert distance model state and configuration.

        Args:
            save_path (str): Path where to save the model (should end in .pt).
        """
        state = {
            "online_1": self.online_1.state_dict(),
            "online_2": self.online_2.state_dict(),
            "target_1": self.target_1.state_dict(),
            "target_2": self.target_2.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "config": self.config,
        }
        torch.save(state, save_path)

    @staticmethod
    def load(load_path: str) -> "HilbertDistance":
        """
        Load the Hilbert Distance model state and configuration without instantiating the class.

        Args:
            load_path (str): Path to the saved model file.

        Returns:
            HilbertDistance: An instance of the HilbertDistance class with the loaded state and configuration.
        """
        state = torch.load(load_path, weights_only=False)

        config = state["config"]
        obs_size = config["obs_size"]
        embedding_size = config["embedding_dim"]
        device = config["device"]

        hilbert_kwargs = {}
        if "hidden_dims" in config:
            hilbert_kwargs["hidden_dims"] = config["hidden_dims"]
        net_1 = HilbertEmbeddingModel(obs_size, embedding_size, **hilbert_kwargs).to(device)
        net_2 = HilbertEmbeddingModel(obs_size, embedding_size, **hilbert_kwargs).to(device)

        instance = HilbertDistance(config=config, network_1=net_1, network_2=net_2)
        instance.online_1.load_state_dict(state["online_1"])
        instance.online_2.load_state_dict(state["online_2"])
        instance.target_1.load_state_dict(state["target_1"])
        instance.target_2.load_state_dict(state["target_2"])
        instance.optimiser.load_state_dict(state["optimiser_state"])

        return instance
