import numpy as np
from Agents.GoalAgent import GoalAgent
from Planning.PlanningEnvModel import PlanningEnvModel
import torch
import torch.nn.functional as F
from Distances import Distance
from typing import Optional, Tuple
from Utils.Utils import infer_distance_device


class PlanningAgent(GoalAgent[np.ndarray, np.ndarray]):
    def __init__(self,
                 distance: Distance,
                 model: PlanningEnvModel[np.ndarray, np.ndarray],
                 action_bins: int,
                 lookahead: int,
                 num_samples: int,
                 smoothing_window: int,
                 action_repeat: int,
                 step_penalty: float = 0.1) -> None:
        """
        Initialize the PlanningAgent.

        Args:
            distance: A function to calculate the distance between two states.
            model: An instance of PlanningEnvModel to predict the next state.
            action_bins: The number of bins to discretize the action space.
            lookahead: The number of steps to look ahead in the planning.
            num_samples: The number of action sequences to sample for planning.
            smoothing_window: The window size for moving average smoothing of distances.
            action_repeat: The minimum number of times an action should be repeated.
        """
        self.distance = distance
        self.model = model
        self._device = infer_distance_device(distance)
        self.action_space = model.sim_env.action_space
        # Discretize the action space
        action_dim = self.action_space.shape[0]
        num_bins_per_dim = int(round(action_bins ** (1 / action_dim)))
        action_ranges = [np.linspace(self.action_space.low[i], self.action_space.high[i], num_bins_per_dim) for i in
                         range(action_dim)]
        self.actions = np.array(np.meshgrid(*action_ranges)).T.reshape(-1, action_dim)
        self.lookahead = lookahead

        # Detect batched model (OGBenchVecEnvModel)
        self._use_batched = hasattr(model, 'predict_batch') and hasattr(model, '_num_envs')
        if self._use_batched:
            self._model_batch_size = model._num_envs
            if num_samples % self._model_batch_size != 0:
                num_samples = max(self._model_batch_size,
                                  (num_samples // self._model_batch_size) * self._model_batch_size)
                print(f"Adjusted num_samples to {num_samples} (multiple of batch size {self._model_batch_size})")

        self.num_samples = num_samples

        self.plan = None
        self.min_distance = float('inf')
        self.smoothing_window = smoothing_window
        self.action_repeat = min(action_repeat, lookahead)
        self.step_penalty = step_penalty
        self.steps_taken = 0

    def _rollout_batched(self, state: np.ndarray, actual_actions: np.ndarray) -> np.ndarray:
        """Roll out trajectories using batched VecEnvModel.

        Args:
            state: Current state (1-D).
            actual_actions: (num_samples, lookahead, action_dim) array of actual actions.

        Returns:
            all_states: (num_samples, lookahead, state_dim) array.
        """
        state_dim = state.shape[0]
        all_states = np.zeros((self.num_samples, self.lookahead, state_dim))

        for start in range(0, self.num_samples, self._model_batch_size):
            end = start + self._model_batch_size
            batch_actions = actual_actions[start:end]  # (batch, lookahead, action_dim)

            current_states = np.tile(state, (self._model_batch_size, 1))
            has_terminated = np.zeros(self._model_batch_size, dtype=bool)

            for t in range(self.lookahead):
                if t == 0:
                    next_obs, _, dones = self.model.predict_batch(
                        current_states, batch_actions[:, t, :])
                else:
                    next_obs, _, dones = self.model.step_batch(
                        batch_actions[:, t, :])

                was_alive = ~has_terminated
                should_update = was_alive & ~dones.astype(bool)
                if np.any(should_update):
                    current_states[should_update] = next_obs[should_update]

                has_terminated |= dones.astype(bool)
                all_states[start:end, t, :] = current_states

        return all_states

    def _plan_with_intermediate_states(self, state: np.ndarray, goal_tensor: torch.Tensor) -> Tuple[
        list, np.ndarray, float]:
        """New planning method: evaluates all intermediate states."""
        num_action_chunks = self.lookahead // self.action_repeat
        action_chunks = np.random.choice(len(self.actions), (self.num_samples, num_action_chunks))

        # Construct full action sequences with repeats and remainder handling
        base_sequences = np.repeat(action_chunks, self.action_repeat, axis=1)
        remainder = self.lookahead % self.action_repeat
        if remainder > 0:
            last_actions = np.repeat(action_chunks[:, -1:], remainder, axis=1)
            action_sequences = np.concatenate([base_sequences, last_actions], axis=1)[:, :self.lookahead]
        else:
            action_sequences = base_sequences

        if self._use_batched:
            # Convert action indices to actual actions for batched rollout
            actual_actions = self.actions[action_sequences]  # (num_samples, lookahead, action_dim)
            all_states = self._rollout_batched(state, actual_actions)
        else:
            all_states = np.zeros((self.num_samples, self.lookahead, state.shape[0]))
            for i, action_sequence in enumerate(action_sequences):
                states_list, _ = self.model.predict_action_sequence([self.actions[action_idx] for action_idx in action_sequence])
                all_states[i, :len(states_list), :] = np.array(states_list)

        next_states_flat = all_states.reshape(-1, state.shape[0])
        next_states_tensor = torch.from_numpy(next_states_flat).float()
        # Process distances in batches to avoid memory issues
        batch_size = 4096
        num_total_states = next_states_tensor.shape[0]
        all_distances = []

        for i in range(0, num_total_states, batch_size):
            batch_end = min(i + batch_size, num_total_states)
            batch_states = next_states_tensor[i:batch_end].to(self._device)

            current_batch_size = batch_states.shape[0]
            batch_goal = goal_tensor.repeat(current_batch_size, 1).to(self._device)

            if hasattr(self.distance, 'eval_dist'):
                distances_batch = self.distance.eval_dist(batch_states, batch_goal)
            else:
                distances_batch = self.distance(batch_states, batch_goal)

            all_distances.append(distances_batch.cpu())

        distances_flat = torch.cat(all_distances)
        distances = distances_flat.view(self.num_samples, self.lookahead)

        # Apply moving average smoothing to find the best trajectory ***
        if self.smoothing_window > 1:
            # Reshape for conv1d: (N, C_in, L_in) -> (num_samples, 1, lookahead)
            distances_for_conv = distances.unsqueeze(1)
            # Create the moving average kernel
            kernel = torch.ones(1, 1, self.smoothing_window, device=distances.device) / self.smoothing_window
            # Apply convolution with padding to keep the output length the same
            padding_amount = (self.smoothing_window - 1) // 2
            # 1. Manually pad the tensor by replicating the edge values
            padded_distances = F.pad(distances_for_conv, (padding_amount, padding_amount), mode='replicate')
            # 2. Apply convolution with NO padding, since we already did it manually
            distances_to_evaluate = F.conv1d(padded_distances, kernel, padding=0).squeeze(1)
        else:
            distances_to_evaluate = distances


        step_indices = torch.arange(self.lookahead, device=distances_to_evaluate.device).float()
        # The cost is the distance plus a penalty for each step taken.
        costs = distances_to_evaluate + self.step_penalty * step_indices.unsqueeze(0)

        # For each sample, find the step with the minimum cost
        min_costs_per_sample, best_step_indices = torch.min(costs, dim=1)

        # Find the sample that has the overall minimum cost
        best_sample_idx = torch.argmin(min_costs_per_sample)
        best_step_idx = best_step_indices[best_sample_idx]

        min_distance = min_costs_per_sample[best_sample_idx].item()
        best_plan = action_sequences[best_sample_idx, :max(best_step_idx + 1, self.action_repeat)].tolist()
        best_next_state = all_states[best_sample_idx, best_step_idx]

        return best_plan, best_next_state, min_distance

    def act(self, state: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Select the action that minimizes the distance to the goal.

        Returns:
            (action, predicted_state) where predicted_state is always None
            for this agent (rollouts are internal to planning).
        """
        goal_tensor = torch.from_numpy(goal).float().unsqueeze(0)

        self.steps_taken += 1
        if not self.plan or self.steps_taken > self.action_repeat:
            best_plan, best_next_state, min_distance = self._plan_with_intermediate_states(state, goal_tensor=goal_tensor)
            if not self.plan or min_distance < self.min_distance:
                self.min_distance = min_distance
                self.plan = best_plan
            self.steps_taken = 0

        best_action_index = self.plan.pop(0)
        best_action = self.actions[best_action_index]

        return best_action, None

    def reset(self) -> None:
        self.plan = None
        self.min_distance = float('inf')
        self.steps_taken = 0

