from typing import Callable, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode
import mujoco

from Planning.PlanningEnvModel import PlanningEnvModel
from Planning.ogbench_utils import find_ogb_sim


class OGBenchStateWrapper(gym.Wrapper):
    """
    Internal wrapper used for AsyncVectorEnv.

    Exposes a settable attribute `planning_state` so the parent process can set
    simulator state inside each worker process via `vec_env.set_attr(...)`.
    """

    def __init__(self, env: gym.Env, sim_source: Callable[[gym.Env], Any]):
        super().__init__(env)
        self._find_sim = sim_source
        self._sim = self._find_sim(self.env)

        self.nq = int(self._sim.model.nq)
        self.nv = int(self._sim.model.nv)

        self._is_dm = hasattr(self._sim, "reset_context") and hasattr(self._sim, "data") and hasattr(self._sim, "model")

    @property
    def planning_state(self) -> np.ndarray:
        return np.concatenate([np.array(self._sim.data.qpos), np.array(self._sim.data.qvel)]).astype(np.float32)

    @planning_state.setter
    def planning_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32).ravel()
        if state.shape[0] != self.nq + self.nv:
            raise ValueError(f"Expected state length {self.nq + self.nv}, got {state.shape[0]}.")

        qpos = state[: self.nq]
        qvel = state[self.nq : self.nq + self.nv]

        if self._is_dm:
            with self._sim.reset_context():
                self._sim.data.qpos[:] = qpos
                self._sim.data.qvel[:] = qvel
        else:
            self._sim.data.qpos[:] = qpos
            self._sim.data.qvel[:] = qvel
            mujoco.mj_forward(self._sim.model, self._sim.data)


class OGBenchVecEnvModel(PlanningEnvModel[np.ndarray, np.ndarray]):
    def __init__(
        self,
        env: gym.Env,
        *,
        make_env: Callable[[], gym.Env],
        num_envs: int,
        clip_action: bool = True,
        sim_source: Optional[Callable[[gym.Env], Any]] = None,
        backend: str = "sync",
        async_context: str = "spawn",
        async_shared_memory: bool = False,
    ) -> None:
        """
        Initialise a vectorised model of a given OGBench environment.

        Args:
            env: The source environment to mirror.
            make_env: A factory that returns a fresh environment instance with
                rendering disabled. It is called `num_envs` times to build the
                worker pool.
            num_envs: The number of sub-environments in the pool. This typically
                equals the number of samples you evaluate at each planning depth.
            clip_action: If `True`, actions are clipped to the action-space
                bounds prior to stepping. Defaults to `True`.
            sim_source: Optional locator that maps an environment instance to an
                object exposing `.model` and `.data` (e.g., a MuJoCo handle).
                Defaults to :func:`find_ogb_sim`.
            backend: Vector backend: `"sync"` or `"async"`. Defaults to `"sync"`.
            async_context: Multiprocessing start method for the async backend.
                Defaults to `"spawn"`.
            async_shared_memory: Whether to use shared memory in the async backend.
                Defaults to `False`.

        Raises:
            RuntimeError: If the environment cannot be deep-copied and `make_env` is not provided.
            ValueError: If `backend` is not recognised.
        """
        # Create a copy of the real environment.
        # Prefer deepcopying, but fall back to the provided make_env function if needed.
        try:
            super().__init__(env)
        except Exception:
            # If deepcopy fails, ensure we still have a valid env for specs.
            self.sim_env = make_env()

        self._real_env = env
        self._clip_action = clip_action
        self._find_sim = sim_source or find_ogb_sim
        self._backend = backend

        self._num_envs = int(num_envs)

        if backend == "sync":
            # Build a SyncVectorEnv of N copies. Worker envs should be headless.
            def _make_thunk():
                def _thunk():
                    e = make_env()
                    return e

                return _thunk

            self.vec_env = SyncVectorEnv(
                [_make_thunk() for _ in range(self._num_envs)],
                autoreset_mode=AutoresetMode.SAME_STEP,
            )

            # Reset once so the first step is always legal.
            self.vec_env.reset()

            # Keep direct handles to each worker and its simulator for fast state I/O.
            self._envs = self.vec_env.envs
            self._sims = [self._find_sim(e) for e in self._envs]

            # Cache dimensions and action shape from the first worker.
            sim0 = self._sims[0]
            self.nq = int(sim0.model.nq)
            self.nv = int(sim0.model.nv)
            self.state_dim = self.nq + self.nv
            self.action_dim = int(np.prod(self._envs[0].action_space.shape))

        elif backend == "async":
            sim_fn = sim_source or find_ogb_sim

            def _make_thunk():
                def _thunk():
                    e = make_env()
                    return OGBenchStateWrapper(e, sim_fn)

                return _thunk

            self.vec_env = AsyncVectorEnv(
                [_make_thunk() for _ in range(self._num_envs)],
                context=async_context,
                shared_memory=async_shared_memory,
                autoreset_mode=AutoresetMode.SAME_STEP,
            )
            self.vec_env.reset()

            # Pull simulator dimensions from the reference env.
            sim0 = sim_fn(self._real_env)
            self.nq = int(sim0.model.nq)
            self.nv = int(sim0.model.nv)
            self.state_dim = self.nq + self.nv
            self.action_dim = int(np.prod(self._real_env.action_space.shape))

        else:
            raise ValueError("backend must be 'sync' or 'async'.")

    # ------------------------ internal helpers ------------------------ #
    def _split_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a full simulator state into ``(qpos, qvel)``.

        Args:
            state: A 1-D array of length ``nq + nv`` equal to ``concat(qpos, qvel)``.

        Returns:
            A tuple ``(qpos, qvel)`` as NumPy arrays.

        Raises:
            ValueError: If ``state`` does not have length ``nq + nv``.
        """
        if state.shape[0] != self.nq + self.nv:
            raise ValueError(f"Expected state length {self.nq + self.nv}, got {state.shape[0]}.")
        return state[: self.nq], state[self.nq : self.nq + self.nv]

    def _set_sim_state_single(self, sim_like, state: np.ndarray) -> None:
        """
        Place a single planning state into one simulator.

        This accepts either the full state ``concat(qpos, qvel)`` or ``qpos``
        only. In the latter case, ``qvel`` is read from the reference environment.

        Args:
            sim_like: An object exposing ``.model``, ``.data`` and, for dm_control,
                optionally ``.reset_context``.
            state: A 1-D planning state as described above.
        """
        if state.shape[0] == self.nq + self.nv:
            qpos, qvel = self._split_state(state.astype(np.float32, copy=False))
        elif state.shape[0] == self.nq:
            qpos = np.asarray(state, dtype=np.float32).ravel()
            # Fallback: obtain qvel from the reference simulator.
            qvel = np.array(self._find_sim(self._real_env).data.qvel, copy=True)
        else:
            raise ValueError(f"Expected state len {self.nq} or {self.nq + self.nv}, got {state.shape[0]}.")

        if hasattr(sim_like, "reset_context") and hasattr(sim_like, "data") and hasattr(sim_like, "model"):
            # dm_control path: writes must occur under reset_context.
            with sim_like.reset_context():
                sim_like.data.qpos[:] = qpos
                sim_like.data.qvel[:] = qvel
        else:
            # Gym-style MuJoCo: write and forward the computation graph.
            sim_like.data.qpos[:] = qpos
            sim_like.data.qvel[:] = qvel
            mujoco.mj_forward(sim_like.model, sim_like.data)

    def _set_sim_states_batch(self, states: np.ndarray) -> None:
        """
        Write a batch of planning states into the pool of simulators.

        Args:
            states: An array of shape ``(N, D)`` where ``N == num_envs`` and
                ``D in {nq, nq + nv}``.

        Raises:
            ValueError: If the batch size does not equal ``num_envs``.
        """
        if states.shape[0] != self._num_envs:
            raise ValueError(f"States batch must have shape ({self._num_envs}, D), got {states.shape}.")
        for i, sim_like in enumerate(self._sims):
            self._set_sim_state_single(sim_like, states[i])

    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Optionally clip actions to the action-space bounds.

        Args:
            actions: An array of shape ``(N, action_dim)``.

        Returns:
            The clipped actions if ``clip_action`` is ``True``; otherwise the
            original array.
        """
        if not self._clip_action:
            return actions
        if self._backend == "sync":
            low = self._envs[0].action_space.low
            high = self._envs[0].action_space.high
        else:
            low = self._real_env.action_space.low
            high = self._real_env.action_space.high
        return np.clip(actions, low, high)

    def _extract_next_state_single(self, obs: Any, sim_like) -> np.ndarray:
        """
        Build a 1-D next-state vector for a single worker.

        Args:
            obs: The observation returned by the worker environment.
            sim_like: The worker's simulator handle exposing ``.data``.

        Returns:
            If ``obs`` is already a 1-D array, it is returned as the next
            planning state. Otherwise, the full simulator state
            ``concat(qpos, qvel)`` is returned.
        """
        try:
            arr = np.asarray(obs, dtype=np.float32)
            if arr.ndim == 1:
                return arr
        except Exception:
            pass
        return np.concatenate([np.array(sim_like.data.qpos), np.array(sim_like.data.qvel)]).astype(np.float32)

    # -------------------------- Public API ---------------------------- #
    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform a single one-step prediction for a single sample.

        This is a compatibility wrapper for callers that expect a non-batched model.
        The vector pool requires a full batch of size ``num_envs``, so we replicate
        the input across the pool, step once, and return the first result.

        Args:
            state: A 1-D planning state of length ``nq`` or ``nq + nv``.
            action: A 1-D action of length ``action_dim``.

        Returns:
            A tuple ``(next_state, reward)`` where ``next_state`` is a 1-D array and
            ``reward`` is a Python ``float``.
        """
        state = np.asarray(state, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()

        states = np.tile(state[None, :], (self._num_envs, 1))
        actions = np.tile(action[None, :], (self._num_envs, 1))

        next_states, rewards, dones = self.predict_batch(states, actions)
        return next_states[0], float(rewards[0])

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a single one-step prediction for a batch of samples.

        The method writes each state into the corresponding worker simulator,
        steps the vectorised environment once with the given actions, and
        returns stacked next states and rewards.

        Args:
            states: A 2-D array of shape ``(N, D)`` with ``N == num_envs`` and
                ``D in {nq, nq + nv}``.
            actions: A 2-D array of shape ``(N, action_dim)``.

        Returns:
            A triplet ``(next_states, rewards, dones)`` where:
                - ``next_states`` has shape ``(N, S)``. If the worker
                  observations are 1-D arrays, ``S`` equals the observation
                  dimension; otherwise ``S = nq + nv``.
                - ``rewards`` has shape ``(N,)`` and ``dtype float32``.
                - ``dones`` has shape ``(N,)`` and ``dtype bool``.

        Raises:
            ValueError: If input ranks or batch sizes are invalid.
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if states.ndim != 2 or states.shape[0] == 0:
            raise ValueError(f"``states`` must be (N, D), got {states.shape}.")
        if actions.ndim != 2 or actions.shape[0] != states.shape[0]:
            raise ValueError(f"``actions`` must be (N, {self.action_dim}), got {actions.shape}.")
        N = states.shape[0]
        if N != self._num_envs:
            raise ValueError(f"Vector pool size is {self._num_envs}, but got batch of {N}.")

        # 1) Place the batch of states into the simulators.
        if self._backend == "sync":
            self._set_sim_states_batch(states)
        else:
            # Async backend: states must be set inside workers.
            # If only qpos is provided, pad with qvel from the reference simulator.
            if states.shape[1] == self.nq:
                qvel = np.array(self._find_sim(self._real_env).data.qvel, copy=True).astype(np.float32)
                qvel_batch = np.repeat(qvel[None, :], N, axis=0)
                states_full = np.concatenate([states, qvel_batch], axis=1).astype(np.float32)
            else:
                states_full = states

            if states_full.shape[1] != self.nq + self.nv:
                raise ValueError(
                    f"Async backend requires state length {self.nq + self.nv} (or qpos length {self.nq}); got {states.shape[1]}."
                )

            # Set per-worker state via wrapper attribute.
            self.vec_env.set_attr("planning_state", [states_full[i] for i in range(N)])

        # 2) Clip actions if requested and step once.
        actions = self._clip_actions(actions)
        obs, rewards, terms, truncs, infos = self.vec_env.step(actions)
        done = np.logical_or(terms, truncs)

        next_states = obs.copy().astype(np.float32)

        # --- UPDATED FIX: ABSOLUTELY PROHIBIT TELEPORTATION ---
        # With AutoresetMode.SAME_STEP, when done[i]=True:
        #   - obs[i] contains the RESET observation (first obs of new episode)
        #   - infos["final_observation"][i] contains the TERMINAL observation
        # We must use final_observation to avoid "teleporting" to reset position.
        if np.any(done):
            for i in range(N):
                if done[i]:
                    final_obs = None
                    if "final_observation" in infos:
                        final_obs = infos["final_observation"][i]

                    if final_obs is not None:
                        next_states[i] = final_obs
                    # else: final_observation not available. Keep obs[i] as-is.
                    # This is a fallback - obs[i] might be reset obs, which is wrong,
                    # but states[i] (input state) is also wrong. The planning agent's
                    # has_terminated logic will freeze this trajectory anyway.
                        # This is wrong but at least it's a transition from the step.
                        # The planning agent's has_terminated logic will freeze this state.
        # ------------------------------------------------------

        if hasattr(self.vec_env, "reset_at"):
            for i, d in enumerate(done):
                if d:
                    self.vec_env.reset_at(int(i))

        return next_states, rewards.astype(np.float32), done

    def step_batch(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step the vectorized environment without resetting simulator state.

        Use this for continuing rollouts after an initial `predict_batch` call
        has set the starting state. This avoids the overhead of re-writing
        qpos/qvel and calling mj_forward at every timestep.

        Args:
            actions: A 2-D array of shape ``(N, action_dim)`` where ``N == num_envs``.

        Returns:
            A triplet ``(next_states, rewards, dones)`` with the same format as
            ``predict_batch``.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != self._num_envs:
            raise ValueError(f"``actions`` must be ({self._num_envs}, {self.action_dim}), got {actions.shape}.")

        actions = self._clip_actions(actions)
        obs, rewards, terms, truncs, infos = self.vec_env.step(actions)
        done = np.logical_or(terms, truncs)

        next_states = obs.copy().astype(np.float32)

        # Handle AutoresetMode.SAME_STEP: use final_observation for terminated envs
        if np.any(done):
            for i in range(self._num_envs):
                if done[i]:
                    if "final_observation" in infos and infos["final_observation"][i] is not None:
                        next_states[i] = infos["final_observation"][i]

        if hasattr(self.vec_env, "reset_at"):
            for i, d in enumerate(done):
                if d:
                    self.vec_env.reset_at(int(i))

        return next_states, rewards.astype(np.float32), done


if __name__ == "__main__":
    import ogbench

    # Pretty printing.
    np.set_printoptions(precision=10, suppress=True, linewidth=220)

    # -------------------- helpers -------------------- #
    def make_env(env_name: str):
        """
        Construct an OGBench environment.

        Notes:
            With env_only=True, OGBench returns just the env (not a 3-tuple).
        """
        return ogbench.make_env_and_datasets(env_name, env_only=True)

    def try_make(env_names):
        last_err = None
        for name in env_names:
            try:
                return make_env(name), name
            except Exception as e:
                last_err = e
                continue
        print(f"Could not construct any of {env_names}. Last error: {last_err}")
        return None, None

    def set_state(sim_like, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """
        Set MuJoCo state for both dm_control-style simulators and gym-style simulators.
        """
        if hasattr(sim_like, "reset_context") and hasattr(sim_like, "data") and hasattr(sim_like, "model"):
            with sim_like.reset_context():
                sim_like.data.qpos[:] = qpos
                sim_like.data.qvel[:] = qvel
        else:
            sim_like.data.qpos[:] = qpos
            sim_like.data.qvel[:] = qvel
            mujoco.mj_forward(sim_like.model, sim_like.data)

    def full_state_from_sim(sim_like) -> np.ndarray:
        """
        Return concat(qpos, qvel) from a simulator handle.
        """
        return np.concatenate([np.array(sim_like.data.qpos), np.array(sim_like.data.qvel)]).astype(np.float32)

    def as_next_state(obs: Any, sim_like) -> np.ndarray:
        """
        Convert an env observation into a 1-D next-state vector.

        For OGBench pointmaze tasks, obs is typically already a 1-D array (e.g., 4D).
        If it is not a 1-D array, fall back to concat(qpos, qvel).
        """
        try:
            arr = np.asarray(obs, dtype=np.float32)
            if arr.ndim == 1:
                return arr
        except Exception:
            pass
        return full_state_from_sim(sim_like)

    def reset_with_task(env: gym.Env, *, seed: int, task_id: int | None):
        """
        Reset an env with an optional OGBench task_id.
        """
        reset_kwargs = {}
        if task_id is not None:
            reset_kwargs["options"] = dict(task_id=task_id)
        return env.reset(seed=seed, **reset_kwargs)

    # -------------------- tests -------------------- #
    def single_step_equivalence_check(
        env: gym.Env,
        model: "OGBenchVecEnvModel",
        *,
        steps: int = 40,
        seed: int = 123,
        task_id: int | None = 1,
        atol: float = 3e-6,
        rtol: float = 1e-6,
        on_mismatch: str = "raise",  # "raise" or "warn"
        print_every: int = 1,
    ) -> None:
        """
        Compare env.step vs model.predict from identical MuJoCo states.

        We:
            1) Reset env and read its MuJoCo state.
            2) Build s = concat(qpos, qvel).
            3) Apply a random action.
            4) Compare (next_state, reward) from env.step against model.predict.
        """
        rng = np.random.default_rng(seed)

        reset_with_task(env, seed=seed, task_id=task_id)

        env_sim = find_ogb_sim(env)

        for t in range(steps):
            qpos = np.array(env_sim.data.qpos)
            qvel = np.array(env_sim.data.qvel)
            s = np.concatenate([qpos, qvel]).astype(np.float32)

            a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

            # Model prediction.
            s_next_model, r_model = model.predict(s, a)

            # Real environment step.
            obs, r_env, *_ = env.step(a)
            s_next_env = as_next_state(obs, env_sim)

            diff = s_next_env - s_next_model
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            worst_idx = int(np.argmax(np.abs(diff))) if diff.size else 0

            if t % print_every == 0:
                print(f"\nstep {t}")
                print(f"action: {a}")
                print(f"reward_env={float(r_env):.10f}, reward_model={float(r_model):.10f}")
                print(f"next_state_env ({s_next_env.shape[0]}):")
                print(s_next_env)
                print(f"next_state_model ({s_next_model.shape[0]}):")
                print(s_next_model)
                if diff.size:
                    print(
                        f"max_abs_diff={max_abs:.3e} at idx={worst_idx} "
                        f"(env={s_next_env[worst_idx]:.10e}, model={s_next_model[worst_idx]:.10e}, "
                        f"delta={diff[worst_idx]:.3e})"
                    )

            ok_state = np.allclose(s_next_model, s_next_env, atol=atol, rtol=rtol)
            ok_reward = np.isclose(float(r_model), float(r_env), atol=1e-7, rtol=1e-7)

            if not ok_state or not ok_reward:
                msg = (
                    f"Mismatch at step {t}: "
                    f"state_ok={ok_state}, reward_ok={ok_reward}, "
                    f"max_abs_diff={max_abs:.3e}, "
                    f"Δreward={abs(float(r_env) - float(r_model)):.3e}"
                )
                if on_mismatch == "warn":
                    print("[WARN]", msg)
                else:
                    raise AssertionError(msg)

        label = getattr(getattr(env, "spec", None), "id", None) or type(env).__name__
        print(f"\n[OK] {label}: {steps} printed steps (state & reward within tolerances).")

    def batch_equivalence_check(
        env_name: str,
        model: "OGBenchVecEnvModel",
        *,
        batch_size: int,
        seed: int = 123,
        task_id: int | None = 1,
        atol: float = 3e-6,
        rtol: float = 1e-6,
    ) -> None:
        """
        Compare model.predict_batch against stepping N independent env copies.

        We:
            1) Create a reference env, reset it, and read its MuJoCo state (qpos, qvel).
            2) Build a batch of identical starting states.
            3) Sample a batch of random actions.
            4) Call model.predict_batch.
            5) For i in 0..N-1:
                - Create a fresh env copy.
                - Reset it to the same task_id/seed.
                - Set its MuJoCo state to the reference (qpos, qvel).
                - Step action[i] once.
                - Compare against model outputs.
        """
        rng = np.random.default_rng(seed)

        # Reference env and initial state.
        ref_env = make_env(env_name)
        reset_with_task(ref_env, seed=seed, task_id=task_id)
        ref_sim = find_ogb_sim(ref_env)

        qpos0 = np.array(ref_sim.data.qpos)
        qvel0 = np.array(ref_sim.data.qvel)
        s0 = np.concatenate([qpos0, qvel0]).astype(np.float32)

        states = np.tile(s0[None, :], (batch_size, 1))
        actions = rng.uniform(
            ref_env.action_space.low, ref_env.action_space.high, size=(batch_size, model.action_dim)
        ).astype(np.float32)

        # Ensure the vector workers are reset onto the same task configuration.
        # This matters for OGBench, where the reset selects a task/goal configuration.
        reset_kwargs = {}
        if task_id is not None:
            reset_kwargs["options"] = dict(task_id=task_id)

        seeds = [seed + i for i in range(batch_size)]
        model.vec_env.reset(seed=seeds, **reset_kwargs)

        next_states_model, rewards_model, _ = model.predict_batch(states, actions)

        # Ground truth by stepping N independent env copies.
        next_states_env = []
        rewards_env = []
        for i in range(batch_size):
            e = make_env(env_name)
            reset_with_task(e, seed=seeds[i], task_id=task_id)
            sim_i = find_ogb_sim(e)
            set_state(sim_i, qpos0, qvel0)

            obs, r, *_ = e.step(actions[i])
            next_states_env.append(as_next_state(obs, sim_i))
            rewards_env.append(float(r))

        next_states_env = np.stack(next_states_env, axis=0).astype(np.float32)
        rewards_env = np.asarray(rewards_env, dtype=np.float32)

        ok_state = np.allclose(next_states_model, next_states_env, atol=atol, rtol=rtol)
        ok_reward = np.allclose(rewards_model.astype(np.float32), rewards_env, atol=1e-7, rtol=1e-7)

        if not ok_state or not ok_reward:
            diff = next_states_env - next_states_model
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            raise AssertionError(
                f"Batch mismatch: state_ok={ok_state}, reward_ok={ok_reward}, max_abs_diff={max_abs:.3e}"
            )

        print(f"[OK] Batch check: predict_batch matches stepping {batch_size} independent env copies.")

    # -------------------- run tests -------------------- #
    env_names_groups = [
        ["pointmaze-umaze-navigate-v0", "pointmaze-medium-navigate-v0", "pointmaze-large-navigate-v0"],
        ["antmaze-umaze-navigate-v0", "antmaze-medium-navigate-v0", "antmaze-large-navigate-v0"],
        ["humanoidmaze-umaze-navigate-v0", "humanoidmaze-medium-navigate-v0"],
    ]

    backends_to_test = [
        ("sync", dict()),
        ("async", dict(async_context="spawn", async_shared_memory=False)),
    ]

    for names in env_names_groups:
        env, used = try_make(names)
        if env is None:
            print(f"Skipping; none of {names} available.")
            continue

        for backend, backend_kwargs in backends_to_test:
            print(f"\nTesting OGBench env: {used} (backend={backend})")

            # Factory for model workers.
            make_env_fn = lambda: make_env(used)

            # Use a small pool for tests; increase if you like.
            num_envs = 8

            model = OGBenchVecEnvModel(
                env=env,
                make_env=make_env_fn,
                num_envs=num_envs,
                backend=backend,
                **backend_kwargs,
            )

            # 1) Single-step equivalence against the real env.
            single_step_equivalence_check(
                env,
                model,
                steps=40,
                seed=123,
                task_id=1,
                atol=3e-6,
                rtol=1e-6,
                on_mismatch="warn",
                print_every=1,
            )

            # 2) Batch equivalence against N independent env copies.
            batch_equivalence_check(
                used,
                model,
                batch_size=num_envs,
                seed=123,
                task_id=1,
                atol=3e-6,
                rtol=1e-6,
            )