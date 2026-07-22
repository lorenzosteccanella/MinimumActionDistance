import importlib.util
import random

from tqdm import tqdm
import torch
import gymnasium as gym
import numpy as np


def collect_goal_oriented_trajectories(env, seed, n_of_trajectories=1000):
    np.random.seed(seed)
    trajectories = []

    for _ in tqdm(range(n_of_trajectories)):
        trajectory_dicts = []

        # 1. Reset
        s, _ = env.reset()

        # Security check: Ensure we don't start ALREADY at the goal
        # (Optional, but helps dataset quality)
        if np.linalg.norm(s[:3] - s[-3:]) < 0.05:
            continue  # Skip this episode, it's too easy/bugged

        done, truncated = False, False

        while not (done or truncated):
            current_pos = s[:3]
            desired_goal = s[-3:]

            # 2. Action (P-Controller)
            # Clip INDIVIDUAL axes to allow max diagonal speed (L-Infinity logic)
            kp = 10.0
            action_xyz = np.clip((desired_goal - current_pos) * kp, -1.0, 1.0)
            a = np.append(action_xyz, 0.0)  # Gripper

            # 3. Step
            s_, r, done, truncated, _ = env.step(a)

            # 4. Save COMPLETE Transition
            transition = {
                "s": torch.tensor(s, dtype=torch.float32),
                "a": torch.tensor(a, dtype=torch.float32),
                "r": torch.tensor([r], dtype=torch.float32),  # Ensure shape (1,)
                "s_": torch.tensor(s_, dtype=torch.float32),
                "done": torch.tensor([float(done)], dtype=torch.float32),
                "truncated": torch.tensor([float(truncated)], dtype=torch.float32),
            }
            trajectory_dicts.append(transition)

            s = s_

        # Only append if we actually collected steps
        if len(trajectory_dicts) > 0:
            trajectories.append(tuple(trajectory_dicts))

    return trajectories


def collect_random_trajectories(env, seed, n_of_trajectories=100, max_step_per_trajectory=None):
    # Collects trajectories in the form of (s, a)

    np.random.seed(seed)
    env.action_space.seed(seed)

    trajectories = []
    trajectory = []
    imposed_truncation = False

    s, _ = env.reset(seed=seed)

    # Helper to standardize s to list if it's scalar
    if isinstance(s, (int, float, np.number)):
        s = [s]
    for _ in tqdm(range(n_of_trajectories), desc="Collecting trajectories", leave=True):
        if not imposed_truncation:
            s, _ = env.reset()
            if isinstance(s, (int, float, np.number)):
                s = [s]

        while True:
            a = env.action_space.sample()
            s_, r, done, truncated, _ = env.step(a)

            # if s_ is an integer or a float, let's embed it in a list
            if isinstance(s_, (int, float, np.number)):
                s_ = [s_]

            # Handle dictionary observations if not using a wrapper that flattens them
            if hasattr(env, "observation_space") and isinstance(env.observation_space, gym.spaces.Dict):
                # Assuming 'observation' key exists for GoalEnvs
                t_s = torch.as_tensor(s["observation"], dtype=torch.float32)
                t_s_ = torch.as_tensor(s_["observation"], dtype=torch.float32)
            else:
                t_s = torch.as_tensor(s, dtype=torch.float32)
                t_s_ = torch.as_tensor(s_, dtype=torch.float32)

            # If the action is not an array, embed it in a list.
            if not isinstance(a, (list, np.ndarray)):
                a_list = [a]
            else:
                a_list = a

            trajectory.append(
                {
                    "s": t_s,
                    "a": torch.as_tensor(a_list, dtype=torch.float32),
                    "r": torch.as_tensor([r], dtype=torch.float32),
                    "s_": t_s_,
                    "done": torch.as_tensor([done], dtype=torch.float32),
                    "truncated": torch.as_tensor([truncated], dtype=torch.float32),
                }
            )

            s = s_

            if done or truncated:
                imposed_truncation = False
                break

            if max_step_per_trajectory is not None and len(trajectory) >= max_step_per_trajectory:
                imposed_truncation = True
                break

        # Append the trajectory to the list of trajectories
        trajectories.append(tuple(trajectory))
        trajectory.clear()

    return trajectories


def vector_norm(x1, x2):
    return torch.linalg.vector_norm(x1 - x2, dim=-1)


def infer_distance_device(distance) -> torch.device:
    if hasattr(distance, 'config') and 'device' in distance.config:
        return torch.device(distance.config['device'])
    try:
        return next(distance.model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device('cpu')


def load_and_format_ogbench_trajectories(ogbench_dataset: dict, max_trajectories: int) -> list[tuple]:
    """
    Loads a pre-existing ogbench (D4RL-style) dataset and formats it into
    a list of trajectories.

    Args:
        ogbench_dataset: A dictionary of NumPy arrays containing the offline data.
                         Expected keys are 'observations', 'actions',
                         'next_observations', 'terminals'.

    Returns:
        A list of trajectories. Each trajectory is a tuple of dictionaries,
        where each dictionary represents a single timestep (s, a, r, s', ...).
    """
    # The final list that will hold all trajectories
    all_trajectories = []

    # A temporary list to build the current trajectory
    current_trajectory = []

    # Extract the flat NumPy arrays from the dataset dictionary
    print(ogbench_dataset.keys())
    observations = ogbench_dataset["observations"]
    actions = ogbench_dataset["actions"]
    next_observations = ogbench_dataset["next_observations"]

    dones = ogbench_dataset["terminals"]

    num_total_steps = len(observations)

    print("Reformatting ogbench dataset into trajectories...")
    for i in tqdm(range(num_total_steps), desc="Processing steps"):
        # The 'done' flag is simply the value from the 'terminals' array.
        is_done = dones[i]

        is_truncated = dones[i]

        step_data = {
            "s": torch.from_numpy(observations[i]).float(),
            "a": torch.from_numpy(actions[i]).float(),
            "r": torch.FloatTensor([-1]),
            "s_": torch.from_numpy(next_observations[i]).float(),
            "done": torch.FloatTensor([is_done]),
            "truncated": torch.FloatTensor([is_truncated]),
        }

        current_trajectory.append(step_data)

        # An episode ends if 'is_done' is true.
        if is_done:
            all_trajectories.append(tuple(current_trajectory))
            current_trajectory = []

            # Stop if we've reached the maximum number of trajectories
            if len(all_trajectories) >= max_trajectories:
                break

    # In case the dataset file ends without a final terminal flag for the last trajectory
    if current_trajectory:
        all_trajectories.append(tuple(current_trajectory))

    print(f"Dataset successfully formatted into {len(all_trajectories)} trajectories.")
    return all_trajectories


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def init_wandb(config: dict, group: str, name: str) -> None:
    import os
    import wandb

    prefix = os.environ.get("WANDB_PROJECT_PREFIX", "")
    project = f"{prefix}_{config['project']}" if prefix else config["project"]
    wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        group=group,
        name=name,
        job_type=config["job_type"],
        config=config,
        reinit=True,
        mode="online" if config.get("track", True) else "disabled",
    )
