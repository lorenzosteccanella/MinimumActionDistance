"""Planning evaluation script for MAD, TDMAD, HILBERT, and QRL distance metrics.

Supports two planning modes:
  - ogbench_sampling:       OGBench environments + PlanningAgent (random sampling)
  - ogbench_dataset_guided: OGBench environments + DatasetGuidedPlanningAgent

QRL models are pre-trained externally and loaded from {model_name}_seed_{seed}_traced.pt
via torch.jit.load().

Usage:
    python Runs/eval_planning.py Runs/configs/plan_ogbench_dataset_mad.py
    python Runs/eval_planning.py Runs/configs/plan_ogbench_dataset_qrl.py
"""
import argparse
import os

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import torch
import wandb

from Envs.OgbenchAntmaze import OgbenchAntmaze
from Envs.OgbenchPointmaze import OgbenchPointmaze
from Utils import init_wandb, load_config, set_seed
from Utils.Metrics import pearson_correlation, ratio_metric, spearman_correlation


def _make_ogbench_env(env_name: str, render_mode=None):
    if "antmaze" in env_name:
        return OgbenchAntmaze(env_name=env_name, render_mode=render_mode), \
               lambda: OgbenchAntmaze(env_name=env_name)
    elif "pointmaze" in env_name:
        return OgbenchPointmaze(env_name=env_name, render_mode=render_mode), \
               lambda: OgbenchPointmaze(env_name=env_name)
    else:
        raise ValueError(f"Unknown OGBench environment: {env_name!r}")


class TorchScriptDistanceWrapper:
    """Wraps a TorchScript QRL model to expose the standard eval_* interface.

    Handles two TorchScript model layouts:
      - Sub-module layout: model.encoder + model.quasimetric_model
      - Direct method layout: model.eval_embed_state() + model.eval_z_dist()
    """

    def __init__(self, scripted_model):
        self.model = scripted_model
        try:
            self._encoder = scripted_model.encoder
            self._quasimetric_model = scripted_model.quasimetric_model
            self._has_submodules = True
        except (AttributeError, RuntimeError):
            self._has_submodules = False

    def eval(self):
        self.model.eval()
        return self

    def eval_embed_state(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._has_submodules:
                return self._encoder.forward(state.to("cuda"))
            return self.model.eval_embed_state(state.to("cuda"))

    def eval_z_dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._has_submodules:
                return self._quasimetric_model.forward(z1, z2).squeeze(-1)
            return self.model.eval_z_dist(z1, z2)

    def eval_dist(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(s1, s2).squeeze(-1)


def _load_distance(config: dict, seed: int):
    trained_models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained_models"
    )
    dc = config["distance_class"]

    if dc == "qrl":
        model_path = os.path.join(trained_models_dir, config["model_name"] + f"_seed_{seed}_traced.pt")
        print(f"Loading QRL model from: {model_path}")
        scripted_model = torch.jit.load(model_path, map_location="cuda")
        scripted_model.eval()
        return TorchScriptDistanceWrapper(scripted_model)

    model_path = os.path.join(trained_models_dir, config["model_name"] + f"_seed_{seed}.pt")
    print(f"Loading model from: {model_path}")
    if dc == "mad":
        from Distances.MadDist import MadDist
        return MadDist.load(model_path)
    elif dc == "mador":
        from Distances.MadDistOr import MadDistOr
        return MadDistOr.load(model_path)
    elif dc == "tdmad":
        from Distances.TDMadDist import TDMadDist
        return TDMadDist.load(model_path)
    elif dc == "hilbert":
        from Distances.HilbertDist import HilbertDistance
        return HilbertDistance.load(model_path)
    else:
        raise ValueError(f"Unknown distance_class: {dc!r}")


def _log_gt_metrics(distance, env, config: dict, wandb_summary: dict) -> None:
    """Compute distance quality metrics against ground truth and log to wandb."""
    s1, s2, gt_distances = env.gt(max_dist_accuracy=100)
    with torch.no_grad():
        if config["distance_class"] == "qrl":
            s1_c = torch.as_tensor(s1, device="cuda", dtype=torch.float32)
            s2_c = torch.as_tensor(s2, device="cuda", dtype=torch.float32)
            pred_distances = distance.eval_dist(s1_c, s2_c).cpu()
        else:
            pred_distances = distance.eval_dist(s1, s2).cpu()
    gt_tensor = torch.as_tensor(gt_distances, dtype=pred_distances.dtype)
    spearman_corr = spearman_correlation(gt_tensor, pred_distances)
    pearson_corr = pearson_correlation(gt_tensor, pred_distances)
    ratio_cv = ratio_metric(gt_tensor, pred_distances)
    print(f"Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f} | Ratio CV: {ratio_cv:.4f}")
    wandb_summary["spearman_corr_gt"] = spearman_corr
    wandb_summary["pearson_corr_gt"] = pearson_corr
    wandb_summary["ratio_cv_gt"] = ratio_cv


def _render_frame(env, predicted_state=None, current_state=None) -> None:
    frame = env.render()
    if frame is None:
        return
    frame = cv2.resize(frame, (frame.shape[1] * 4, frame.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    if current_state is not None:
        cv2.putText(frame, f"Curr: {current_state[:2].round(2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if predicted_state is not None:
        cv2.putText(frame, f"Pred: {predicted_state[:2].round(2)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("OGBench", frame[:, :, ::-1])
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_ogbench_sampling_episodes(env, agent, config: dict, seed: int) -> list:
    """Episode loop for OGBench environments with sampling-based planning."""
    results = []
    distance_accuracy = {}

    for i in range(config["num_eval_episodes"]):
        obs, info = env.reset(seed=seed)
        s = obs
        goal = info.get("goal")
        s_0 = s.copy()
        tot_reward = 0.0
        tot_distance = 0.0
        initial_dist = round(float(np.linalg.norm(s_0[:2] - goal[:2])), 0)

        agent.reset()
        while True:
            action, _ = agent.act(s, goal)
            obs, _, terminated, truncated, info = env.step(action)
            if config.get("render_mode"):
                _render_frame(env)
            s = obs.astype(np.float32)
            tot_reward -= 1
            tot_distance += np.linalg.norm(s[:2] - s_0[:2])
            if terminated or truncated:
                break

        success = int(info.get("success", False))
        print(f"episode: {i}, reward: {tot_reward:.0f}")
        wandb.log({
            "success": success,
            "episode_reward": tot_reward,
            "distance_traversed": tot_distance,
            "initial_distance_to_goal": initial_dist,
            "episode_num": i,
        })
        distance_accuracy.setdefault(initial_dist, []).append(success)
        results.append({"success": success, "initial_dist": initial_dist})

    _log_distance_accuracy_table(distance_accuracy)
    return results


def _run_ogbench_dataset_guided_episodes(env, agent, config: dict, seed: int) -> list:
    """Episode loop for OGBench with dataset-guided planning and stuck detection."""
    stuck_window = config.get("stuck_window", 50)
    stuck_threshold = config.get("stuck_threshold", 1.0)
    stuck_goal_progress = config.get("stuck_goal_progress", 0.5)
    stuck_patience = config.get("stuck_patience", 3)
    max_steps = config.get("max_episode_steps", 1000)

    results = []
    distance_accuracy = {}

    for i in range(config["num_eval_episodes"]):
        np.random.seed(seed + i)
        obs, info = env.reset(seed=seed + i)
        s = obs
        goal = info.get("goal")
        s_0 = s.copy()
        initial_dist = round(float(np.linalg.norm(s_0[:2] - goal[:2])), 0)

        tot_reward = 0.0
        n_steps = 0
        position_history = []
        dist_to_goal_history = []
        stuck_count = 0
        is_stuck = False

        agent.reset()
        while True:
            n_steps += 1
            action, predicted_final_state = agent.act(s, goal)
            obs, _, terminated, truncated, info = env.step(action)

            current_pos = obs[:2].copy()
            current_dist_to_goal = np.linalg.norm(current_pos - goal[:2])
            position_history.append(current_pos)
            dist_to_goal_history.append(current_dist_to_goal)
            if len(position_history) > stuck_window:
                position_history.pop(0)
                dist_to_goal_history.pop(0)

            if len(position_history) == stuck_window and n_steps % stuck_window == 0:
                positions = np.array(position_history)
                max_movement = np.max(np.linalg.norm(positions - positions[0], axis=1))
                goal_progress = dist_to_goal_history[0] - dist_to_goal_history[-1]
                if (max_movement < stuck_threshold) and (goal_progress < stuck_goal_progress):
                    stuck_count += 1
                    if stuck_count >= stuck_patience:
                        is_stuck = True
                        print(f"  Stuck at step {n_steps}: movement={max_movement:.2f}m, "
                              f"goal_progress={goal_progress:.2f}m")
                else:
                    stuck_count = 0

            if agent.done_sub_traj and predicted_final_state is not None:
                pos_dist = np.linalg.norm(predicted_final_state[:2] - obs[:2])
                if pos_dist > 2.0:
                    print(f"\nWarning: position divergence {pos_dist:.2f}m")

            if config.get("render_mode"):
                _render_frame(env, predicted_state=predicted_final_state, current_state=obs)

            s = obs.astype(np.float32)
            tot_reward -= 1.0

            if terminated or is_stuck or n_steps >= max_steps:
                break

        success = int(info.get("success", False))
        exit_reason = "success" if success else ("stuck" if is_stuck else "timeout")
        print(f"Episode: {i} | Success: {success} | Steps: {n_steps} | Exit: {exit_reason}")
        wandb.log({
            "success": success,
            "episode_reward": tot_reward,
            "initial_distance_to_goal": initial_dist,
            "episode_num": i,
            "exit_reason_stuck": int(is_stuck),
            "steps_to_finish": n_steps,
        })
        distance_accuracy.setdefault(initial_dist, []).append(success)
        results.append({"success": success, "initial_dist": initial_dist})

    cv2.destroyAllWindows()
    return results


def _log_distance_accuracy_table(distance_accuracy: dict) -> None:
    table = wandb.Table(columns=["initial_distance_to_goal", "success_rate"])
    for dist, successes in sorted(distance_accuracy.items()):
        table.add_data(dist, np.mean(successes))
    wandb.log({"distance_accuracy": table})


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(config: dict) -> float:
    seed = config["seed"]
    set_seed(seed)

    print(f"--- Starting evaluation for seed: {seed} ---")

    init_wandb(config, f"{config['exp_name']}/{config['env_name']}/{config.get('model_name', '')}", f"seed_{seed}")

    mode = config["planning_mode"]
    env_name = config["env_name"]

    # --- Setup environment and model ---
    if mode in ("ogbench_sampling", "ogbench_dataset_guided"):
        env, make_env = _make_ogbench_env(env_name, render_mode=config.get("render_mode"))
        from Planning.OGBenchVecEnvModel import OGBenchVecEnvModel
        num_cores = config.get("num_cores", 5)
        planning_model = OGBenchVecEnvModel(env, make_env=make_env,
                                            num_envs=num_cores, backend="sync")
    else:
        raise ValueError(f"Unknown planning_mode: {mode!r}")

    # --- Load distance model ---
    distance = _load_distance(config, seed)
    # QRL models are already on cuda (loaded via torch.jit.load); set eval mode for others
    dc = config["distance_class"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dc != "qrl":
        if hasattr(distance, "model"):
            distance.model.eval()
            distance.model.to(device)

    # --- Log GT correlation metrics ---
    _log_gt_metrics(distance, env, config, wandb.summary)

    # --- Build agent ---
    if mode == "ogbench_sampling":
        from Agents.PlanningAgent import PlanningAgent
        agent = PlanningAgent(
            distance, planning_model,
            action_bins=config["action_bins"],
            lookahead=config["lookahead"],
            num_samples=config["num_samples"],
            smoothing_window=config.get("smoothing_window", 1),
            action_repeat=config["action_repeat"],
        )
    else:  # ogbench_dataset_guided
        from Agents.DatasetGuidedPlanningAgent import DatasetGuidedPlanningAgent
        agent = DatasetGuidedPlanningAgent(
            distance=distance,
            model=planning_model,
            dataset=env.trainset,
            lookahead=config["lookahead"],
            num_samples=config["num_samples"],
            num_iterations=config.get("num_iterations", 1),
            num_elites=config.get("num_elites", 1),
            step_penalty=config.get("step_penalty", 0.01),
            cost_metric=config.get("cost_metric", "last_state"),
            k_neighbors=config.get("k_neighbors", 50),
            similarity_fn=config.get("similarity_fn", None),
        )
        agent.debug = config.get("debug_cem", False)

    # --- Run episodes ---
    if mode == "ogbench_sampling":
        results = _run_ogbench_sampling_episodes(env, agent, config, seed)
    else:
        results = _run_ogbench_dataset_guided_episodes(env, agent, config, seed)

    overall_success_rate = float(np.mean([r["success"] for r in results]))
    wandb.run.summary["mean_success_rate"] = overall_success_rate
    wandb.finish()

    print(f"--- Finished seed {seed}: success rate = {overall_success_rate:.3f} ---")
    return overall_success_rate


def run(runs: list, seeds: list, base_config: dict) -> None:
    """Outer loop over (env_name, model_name) pairs and seeds."""
    for run_cfg in runs:
        env_name = run_cfg["env_name"]
        model_name = run_cfg["model_name"]
        # Per-environment overrides (e.g. lookahead, action_repeat for giant mazes)
        per_env = run_cfg.get("overrides", {})
        print(f"\nEvaluating {env_name} with model {model_name}")

        seed_success_rates = []
        for seed in seeds:
            config = {**base_config, **per_env, "env_name": env_name,
                      "model_name": model_name, "seed": seed}
            success_rate = run_evaluation(config)
            seed_success_rates.append(success_rate)

        mean_sr = np.mean(seed_success_rates)
        std_sr = np.std(seed_success_rates)
        print(f"\n--- Summary for {env_name} ---")
        print(f"Mean success rate: {mean_sr:.4f} ± {std_sr:.4f}")

        if base_config.get("track", True):
            prefix = os.environ.get("WANDB_PROJECT_PREFIX", "")
            project = f"{prefix}_{base_config['project']}" if prefix else base_config["project"]
            wandb.init(
                project=project,
                entity=os.environ.get("WANDB_ENTITY"),
                group=f"{base_config['exp_name']}/summary",
                name=env_name,
                job_type=base_config["job_type"] + "_summary",
                config=base_config,
                reinit=True,
            )
            wandb.summary["mean_success_rate"] = mean_sr
            wandb.summary["std_success_rate"] = std_sr
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a planning agent")
    parser.add_argument("config", help="Path to a config file (see Runs/configs/)")
    args = parser.parse_args()

    mod = load_config(args.config)
    config, runs, seeds = mod.CONFIG, mod.RUNS, mod.SEEDS
    run(runs, seeds, config)
