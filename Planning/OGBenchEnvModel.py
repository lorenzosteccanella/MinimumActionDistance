from typing import Callable, Optional, Tuple, Any
import cv2
import numpy as np
import gymnasium as gym
import mujoco

from Planning.PlanningEnvModel import PlanningEnvModel
from Planning.ogbench_utils import find_ogb_sim

def render_frame(env):
    """Show a frame if the env returns one (RGB -> BGR for OpenCV)."""
    frame = env.render()  # H x W x 3, RGB
    if frame is not None:
        frame = cv2.resize(frame, (frame.shape[1] * 4, frame.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("OGBench", frame[:, :, ::-1])
        cv2.waitKey(1)


class OGBenchEnvModel(PlanningEnvModel[np.ndarray, np.ndarray]):
    def __init__(
        self,
        env: gym.Env,
        *,
        make_env: Optional[Callable[[], gym.Env]] = None,
        sim_source: Optional[Callable[[gym.Env], Any]] = None,
        clip_action: bool = True,
    ) -> None:
        """
        Instantiates a model of a given OGBench environment.

            predict(state, action) -> (next_state, reward)

        `state` can either be `concat(qpos, qvel)` or just `qpos`.

        Args:
            env (gym.Env): The source environment to mirror.
            make_env: Factory used to create a fresh environment if `deepcopy` is not supported by `env`. Required only when deepcopy fails.
            sim_source: A function that takes the environment and returns the base object that exposes the .model, .data, or other simulation attributes. Defaults to None.
            clip_action: Whether to clip actions to the action space. Defaults to True.

        Raises:
            RuntimeError: If the environment cannot be deep-copied and `make_env` is not provided.
        """
        # Create a copy of the real environment.
        # Prefer deepcopying, but fall back to the provided make_env function if needed.
        try:
            super().__init__(env)
        except Exception:
            if make_env is None:
                raise RuntimeError(
                    "deepcopying env failed. Pass a function make_env=... to let the model construct a copy of the environment from scratch."
                )
            self.sim_env = make_env()

        # Keep a reference to the real environment.  # TODO is this a reference to the real environment, such that I can always get the actual qpos and qvel from it?
        self._real_env = env

        # Initialise a copy of the environment to use for forward prediction.
        self.sim_env.reset(seed=42)
        self._sim = (sim_source or find_ogb_sim)(self.sim_env)

        # Detect simulator style (dm_control vs. Gym MuJoCo).
        self._is_dm = hasattr(self._sim, "reset_context") and hasattr(self._sim, "data") and hasattr(self._sim, "model")

        # Cache dimensions and behavior flags.
        self.nq = int(self._sim.model.nq)
        self.nv = int(self._sim.model.nv)
        self.state_dim = self.nq + self.nv
        self.action_dim = int(np.prod(self.sim_env.action_space.shape))
        self._clip_action = clip_action

    def _split_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a full simulator state vector into `(qpos, qvel)`.

        Args:
            state: A 1D array of length `nq + nv` containing
                `concat(qpos, qvel)`.

        Returns:
            A tuple `(qpos, qvel)` as NumPy arrays.

        Raises:
            ValueError: If `state` does not have length `nq + nv`.
        """
        if state.shape[0] != self.nq + self.nv:
            raise ValueError(f"Expected state length {self.nq + self.nv}, got {state.shape[0]}")
        return state[: self.nq], state[self.nq : self.nq + self.nv]

    def _set_sim_state(self, state: np.ndarray) -> None:
        """
        Place the provided planning state into the simulator.

        This method either:
          - uses `state_to_qpos_qvel` (if provided) to translate the planner's
            state to `(qpos, qvel)`, or
          - treats `state` as `concat(qpos, qvel)` and splits it with
            `_split_state`.

        The resulting `(qpos, qvel)` are written into the simulator, using a
        `reset_context()` if required by dm_control, or `mj_forward` otherwise.
        """
        qpos, qvel = self._split_state(state.astype(np.float32, copy=False))

        if self._is_dm:
            # dm_control requires writes under a reset context.
            with self._sim.reset_context():
                self._sim.data.qpos[:] = qpos
                self._sim.data.qvel[:] = qvel
        else:
            # Gym-style: write and forward the computation graph.
            self._sim.data.qpos[:] = qpos
            self._sim.data.qvel[:] = qvel
            mujoco.mj_forward(self._sim.model, self._sim.data)

    def _extract_next_state(self, obs: Any) -> np.ndarray:
        """
        Build the planner's next state after a step.

        Priority order:
            1) If `obs_key` is set and `obs` is a dict, return `obs[obs_key]`.
            2) If `extract_state` is provided, call it with `(obs, model, data)`.
            3) If `obs` is array-like (e.g., OGBench returns a plain ndarray),
               return `np.asarray(obs)`.
            4) Fallback: return `concat(qpos, qvel)` from the simulator.

        Returns:
            A 1D `np.float32` array representing the next planning state.
        """
        try:
            arr = np.asarray(obs, dtype=np.float32)
            if arr.ndim == 1:
                return arr
        except Exception:
            pass

        return np.concatenate([np.array(self._sim.data.qpos), np.array(self._sim.data.qvel)]).astype(np.float32)

    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predicts the next state and reward based on the current state and action.

        This method takes the current state and action as input and predicts the next state
        and reward. The state can either include both position (`qpos`) and velocity (`qvel`)
        components, or only the position component. If only the position component is provided,
        the velocity component is extracted from the environment.

        Args:
            state (np.ndarray): The current state, which can be either:
                - A 1D array of length `nq + nv` (position and velocity components).
                - A 1D array of length `nq` (position component only).
            action (np.ndarray): A 1D array representing the action, with length `action_dim`.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - The next state (or the position component of the next state if only `qpos` was provided).
                - The reward associated with the transition.

        Raises:
            ValueError: If `state` is not a 1D numpy array.
            ValueError: If `action` is not a 1D numpy array.
            ValueError: If the length of `action` does not match `action_dim`.
            ValueError: If the length of `state` is not `nq` or `nq + nv`.
        """
        # Check that the state and action inputs are valid.
        if not isinstance(state, np.ndarray):
            raise ValueError(f"Expected state to be a np.ndarray, got {type(state)}")
        if not isinstance(action, np.ndarray):
            raise ValueError(f"Expected action to be a np.ndarray, got {type(action)}")
        if state.ndim != 1:
            raise ValueError(f"Expected state to be 1D, got {state.ndim}D")
        if action.ndim != 1:
            raise ValueError(f"Expected action to be 1D, got {action.ndim}D")
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action length {self.action_dim}, got {action.shape[0]}")

        # If both qpos and qvel are provided, we can predict directly.
        if state.shape[0] == self.nq + self.nv:
            return self._predict(state, action)
        # If only qpos is provided, we need to extract qvel from the provided env before predicting.
        elif state.shape[0] == self.nq:
            # We need to extract the "qpos" part from the provided state.
            qpos = np.asarray(state, dtype=np.float32).ravel()
            if qpos.size != self.nq:
                raise ValueError(f"Expected state length {self.nq}, got {qpos.size}")

            # We need to extract the "qvel" part from the provided environment.
            qvel = np.array(find_ogb_sim(self._real_env).data.qvel, copy=True)

            # Concatenate qpos and qvel to form the full state.
            full_state = np.concatenate([qpos, qvel]).astype(np.float32)

            # Now we can call the internal predict method with the full state.
            full_next_state, reward = self._predict(full_state, action)

            # We only return the "qpos" part of the next state.
            next_qpos = full_next_state[: self.nq]
            return next_qpos, reward
        # If the state length is neither nq nor nq+nv, we raise an error.
        else:
            raise ValueError(f"Expected state length {self.nq} or {self.nq + self.nv}, got {state.shape[0]}")

    def predict_action_sequence(self, action_sequence: list[np.ndarray]) -> Tuple[list[np.ndarray], list[float]]:
        """
        Predicts the next state and reward based on the current state and action.

        This method takes the current state and action as input and predicts the next state
        and reward. The state can either include both position (`qpos`) and velocity (`qvel`)
        components, or only the position component. If only the position component is provided,
        the velocity component is extracted from the environment.

        Args:
            action_sequence (list[np.ndarray]): A list of 1D arrays representing the action sequence, each with length `action_dim`.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - The next state (or the position component of the next state if only `qpos` was provided).
                - The reward associated with the transition.

        Raises:
            ValueError: If `state` is not a 1D numpy array.
            ValueError: If `action` is not a 1D numpy array.
            ValueError: If the length of `action` does not match `action_dim`.
            ValueError: If the length of `state` is not `nq` or `nq + nv`.
        """


        # real actual state in the environment
        real_env_state = find_ogb_sim(self._real_env)
        qpos = np.array(real_env_state.data.qpos, copy=True)
        qvel = np.array(real_env_state.data.qvel, copy=True)

        # Concatenate qpos and qvel to form the full state.
        full_state = np.concatenate([qpos, qvel]).astype(np.float32)

        states_list = []
        rewards_list = []
        for action in action_sequence:
            # Now we can call the internal predict method with the full state.
            next_state, reward = self._predict(full_state, action)

            sim_env_state = find_ogb_sim(self.sim_env)
            qpos = np.array(sim_env_state.data.qpos, copy=True)
            qvel = np.array(sim_env_state.data.qvel, copy=True)
            full_state = np.concatenate([qpos, qvel]).astype(np.float32)

            # We only return the "qpos" part of the next state
            states_list.append(next_state)
            rewards_list.append(reward)

            #render_frame(self.sim_env)

        assert len(states_list) == len(action_sequence)
        assert len(rewards_list) == len(action_sequence)

        return states_list, rewards_list

    def _predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        # Set the simulator to the provided state.
        self._set_sim_state(state)

        # Clip action if needed.
        a = np.asarray(action, dtype=np.float32).copy()
        if self._clip_action:
            a = np.clip(a, self.sim_env.action_space.low, self.sim_env.action_space.high)

        # Step the simulator, then extract the next-state and reward.
        obs, reward, *_ = self.sim_env.step(a)
        next_state = self._extract_next_state(obs)
        return next_state, float(reward)

    # Convenient alias for predict.
    def model(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.predict(s, a)

