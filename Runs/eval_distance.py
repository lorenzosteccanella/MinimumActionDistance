"""Unified distance evaluation script for MAD, TDMAD, and HILBERT distance metrics.

Usage:
    python Runs/eval_distance.py Runs/configs/dist_mad.py
"""
import argparse
import gc
import os
from collections import deque

from dotenv import load_dotenv
load_dotenv()

import gymnasium as gym
import numpy as np
import torch
import wandb
from tqdm import trange

from Envs import Umaze
from Envs.CliffWalking import NormalizedCliffWalking
from Envs.GridWorlds.EmptyGridWorld import EmptyGridWorld
from Envs.GridWorlds.KeyDoorGridWorld import KeyDoorGridWorld
from Envs.Mediummaze import Mediummaze
from Envs.OgbenchAntmaze import OgbenchAntmaze, OgbenchAntmazeDeterministicTeleport
from Envs.OgbenchPointmaze import OgbenchPointmaze
from Utils import (
    collect_goal_oriented_trajectories,
    collect_random_trajectories,
    init_wandb,
    load_and_format_ogbench_trajectories,
    load_config,
    set_seed,
)
from Utils.Metrics import pearson_correlation, ratio_metric, spearman_correlation



def _make_distance(config: dict):
    dc = config["distance_class"]
    if dc == "mad":
        from Distances.MadDist import MadDist
        return MadDist(config=config)
    elif dc == "tdmad":
        from Distances.TDMadDist import TDMadDist
        return TDMadDist(config=config)
    elif dc == "hilbert":
        from Distances.HilbertDist import HilbertDistance
        from Models.HilbertModels import HilbertEmbeddingModel
        hilbert_kwargs = {}
        if "hidden_dims" in config:
            hilbert_kwargs["hidden_dims"] = config["hidden_dims"]
        net1 = HilbertEmbeddingModel(config["obs_size"], config["embedding_dim"], **hilbert_kwargs)
        net2 = HilbertEmbeddingModel(config["obs_size"], config["embedding_dim"], **hilbert_kwargs)
        return HilbertDistance(config, net1, net2)
    elif dc == "mador":
        from Distances.MadDistOr import MadDistOr
        return MadDistOr(config=config)
    else:
        raise ValueError(f"Unknown distance_class: {dc!r}")


def _train_step(distance) -> tuple:
    """Normalize distance.train(steps=1) to always return (loss_o, loss_c, loss)."""
    result = distance.train(steps=1)
    if isinstance(result, list):
        # HilbertDist returns a list of per-step scalar losses
        loss = float(result[0])
        return loss, 0.0, loss
    loss_o, loss_c, loss = result
    return loss_o, loss_c, loss


def _collect_trajectories(env, seed: int, config: dict) -> list:
    n_traj = config.get("er_max_n_traj", 1000)
    if isinstance(env, (EmptyGridWorld, KeyDoorGridWorld, NormalizedCliffWalking)):
        return collect_random_trajectories(env, seed, n_of_trajectories=min(n_traj, 100))
    elif isinstance(env, (Umaze, Mediummaze)):
        return collect_random_trajectories(env, seed, n_of_trajectories=n_traj)
    elif isinstance(env, OgbenchAntmazeDeterministicTeleport):
        return env.load_curated_trajectories(max_trajectories=n_traj)
    elif isinstance(env, (OgbenchPointmaze, OgbenchAntmaze)):
        return load_and_format_ogbench_trajectories(env.trainset, max_trajectories=n_traj)
    else:
        return collect_random_trajectories(env, seed, n_of_trajectories=n_traj)


def _build_debug_log(env_name: str, step: int, s1, s2, d_gt, pred_dist, ratio_cv) -> dict:
    """Return a wandb log dict with debug HTML. Caller merges into the main log call."""
    nonzero_mask = d_gt > 0
    original_indices = torch.where(nonzero_mask)[0]
    ratios = pred_dist[nonzero_mask] / d_gt[nonzero_mask]
    mean_ratio = ratios.mean()
    ratio_errors = (ratios - mean_ratio).abs()
    top_n = min(200, len(ratio_errors))
    top_idx = original_indices[ratio_errors.argsort(descending=True)[:top_n]]

    lines = [
        "-" * 80,
        f"Overall metrics at step {step}:",
        f"Ratio CV: {ratio_cv:.4f}",
        f"Mean ratio: {mean_ratio:.4f}",
        "-" * 80,
        "Top ratio deviations:",
    ]
    for i in top_idx:
        ratio_idx = torch.where(original_indices == i)[0][0]
        if "CliffWalking" in env_name:
            s1_p = (int(s1[i][0].item() * 11), int(s1[i][1].item() * 3))
            s2_p = (int(s2[i][0].item() * 11), int(s2[i][1].item() * 3))
        elif "EmptyGridWorld" in env_name or "KeyDoor" in env_name:
            s1_p = s1[i] * 13
            s2_p = s2[i] * 13
        else:
            s1_p, s2_p = s1[i], s2[i]
        lines.append(
            f"From {s1_p!r:>15} to {s2_p!r:>15}: "
            f"GT={d_gt[i]:6.2f}, Pred={pred_dist[i]:6.2f}, Ratio={ratios[ratio_idx]:6.2f}"
        )
    return {"debug_output": wandb.Html("<pre>" + "\n".join(lines) + "</pre>")}


def evaluate_environment(env_name: str, distance, s1, s2, d_gt, config: dict) -> dict:
    debug = config.get("debug", False)
    loss_c_mw = deque(maxlen=100)
    loss_o_mw = deque(maxlen=100)
    loss_mw = deque(maxlen=100)
    error_mw = deque(maxlen=100)

    t = trange(config["gradient_steps"], desc=f"Training {env_name}", leave=True)
    for step in t:
        loss_o, loss_c, loss = _train_step(distance)
        loss_o_mw.append(loss_o)
        loss_c_mw.append(loss_c)
        loss_mw.append(loss)

        if step % 200 == 0:
            with torch.no_grad():
                pred_dist = distance.eval_dist(s1, s2).cpu()
                if env_name == "KeyDoor":
                    pred_dist = torch.clamp(pred_dist, max=100)

                errors = (d_gt - pred_dist).abs()
                mean_error = errors.mean().item()
                error_mw.append(mean_error)

                correlation = pearson_correlation(d_gt, pred_dist)
                spearman_corr = spearman_correlation(d_gt, pred_dist)
                ratio_cv = ratio_metric(d_gt, pred_dist)

                # Single log call per step to avoid the "step must be monotonically
                # increasing" wandb warning that occurs with multiple log() calls at
                # the same step value.
                log_dict = {
                    "loss_o": loss_o,
                    "loss_c": loss_c,
                    "loss": loss,
                    "mean_error": mean_error,
                    "max_error": errors.max().item(),
                    "correlation": correlation,
                    "spearman_corr": spearman_corr,
                    "ratio_cv": ratio_cv,
                }
                if debug:
                    log_dict.update(_build_debug_log(env_name, step, s1, s2, d_gt, pred_dist, ratio_cv))
                wandb.log(log_dict, step=step)

                t.set_description(
                    f"{env_name} - loss_o: {np.mean(loss_o_mw):.4f}, "
                    f"loss_c: {np.mean(loss_c_mw):.4f}, "
                    f"loss: {np.mean(loss_mw):.4f}, "
                    f"error: {np.mean(error_mw):.4f}"
                )

    with torch.no_grad():
        final_pred = distance.eval_dist(s1, s2).cpu()
        final_error = (d_gt - final_pred).abs()
        nonzero_mask = d_gt > 1e-6

        # --- Cumulative range metrics [0, k] for increasing k ---
        unique_gt_vals = d_gt[nonzero_mask].unique().sort().values
        all_uppers = [v.item() for v in unique_gt_vals
                      if unique_gt_vals[unique_gt_vals <= v].numel() >= 2]
        if len(all_uppers) > 20:
            step = len(all_uppers) // 20
            cumulative_uppers = all_uppers[step - 1::step][:20]
        else:
            cumulative_uppers = all_uppers

        # Use define_metric so wandb plots cumulative metrics against
        # upper_bound instead of the global training step.  This lets
        # different runs in the same group appear as separate lines.
        wandb.define_metric("cumulative/upper_bound")
        wandb.define_metric("cumulative/*", step_metric="cumulative/upper_bound")

        print(f"\n{'=' * 100}")
        print("Cumulative range metrics [0 : k] — each row includes all pairs with gt <= k:")
        print(f"{'=' * 100}")
        print(f"{'Range':<12} {'N':<7} {'Uniq GT':<9} "
              f"{'ScRE%':<11} {'Pearson':<12} {'Spearman':<12} {'Ratio CV'}")
        print("-" * 100)

        cumulative_rows = []
        for k in cumulative_uppers:
            in_range = nonzero_mask & (d_gt <= k)
            n_samples = in_range.sum().item()
            if n_samples < 2:
                continue

            gt_r = d_gt[in_range]
            pred_r = final_pred[in_range]
            n_unique_gt = gt_r.unique().numel()
            if n_unique_gt < 2:
                continue

            range_scale = pred_r.mean().item() / gt_r.mean().item()
            scaled_err = ((gt_r - pred_r / range_scale).abs() / gt_r * 100).mean().item()
            pearson_r = pearson_correlation(gt_r, pred_r)
            spearman_r = spearman_correlation(gt_r, pred_r)
            cv_r = ratio_metric(gt_r, pred_r).item()

            cumulative_rows.append({
                "upper_bound": k,
                "spearman": spearman_r,
                "pearson": pearson_r,
                "ratio_cv": cv_r,
                "scaled_rel_err": scaled_err,
                "n_samples": n_samples,
            })

            wandb.log({
                "cumulative/upper_bound": k,
                "cumulative/spearman": spearman_r,
                "cumulative/pearson": pearson_r,
                "cumulative/ratio_cv": cv_r,
                "cumulative/scaled_rel_err": scaled_err,
            })
            print(f"0-{k:<9.1f}  {n_samples:<7} {n_unique_gt:<9} "
                  f"{scaled_err:<11.2f} {pearson_r:<12.4f} {spearman_r:<12.4f} {cv_r:<10.4f}")

        print("=" * 100)

        return {
            "mean_error": final_error.mean().item(),
            "max_error": final_error.max().item(),
            "std_error": final_error.std().item(),
            "correlation": torch.corrcoef(torch.stack([d_gt, final_pred]))[0, 1].item(),
            "spearman_corr": spearman_correlation(d_gt, final_pred),
            "ratio_cv": ratio_metric(d_gt, final_pred).item(),
            "cumulative": cumulative_rows,
        }


def evaluate_single_seed(env_name: str, env_factory, seed: int, config: dict) -> dict:
    set_seed(seed)

    env = env_factory()
    trajectories = _collect_trajectories(env, seed, config)

    # Copy config and set input dimension from environment
    config = dict(config)
    obs_size = int(np.prod(env.observation_space.shape))
    config["in_d"] = obs_size

    if config["distance_class"] == "hilbert":
        config["obs_size"] = obs_size

    distance = _make_distance(config)
    distance.add_trajectories(trajectories)

    if config["distance_class"] == "hilbert":
        distance.train(steps=0, process_her_trajectories=True, verbose=True)

    max_acc = config.get("max_dist_accuracy", 100)
    s1, s2, d_gt = env.gt(max_dist_accuracy=max_acc)

    metrics = evaluate_environment(env_name, distance, s1, s2, d_gt, config)

    model_save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "trained_models",
        f"{config['exp_name']}_{env_name}_seed_{seed}.pt",
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    distance.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    if isinstance(env, gym.Env):
        env.close()

    # Release GPU memory before the next seed starts.
    # Explicitly delete the HER buffer first — TorchRL's LazyTensorStorage has
    # internal reference cycles that Python's refcount alone cannot break.
    if hasattr(distance, 'her_buffer'):
        del distance.her_buffer
    del distance, trajectories, s1, s2, d_gt
    gc.collect()           # breaks any remaining reference cycles
    torch.cuda.empty_cache()

    return metrics


def evaluate(seeds: list, environments: list, config: dict) -> None:
    metrics_keys = ["mean_error", "max_error", "std_error", "correlation", "spearman_corr", "ratio_cv"]
    results = {}

    for env_name, env_factory in environments:
        env_results = []
        for seed in seeds:
            init_wandb(config, f"{config['exp_name']}/{env_name}", f"seed_{seed}")

            print(f"\nEvaluating {env_name} with seed {seed}")
            metrics = evaluate_single_seed(env_name, env_factory, seed, config)
            env_results.append({**metrics, "seed": seed})

            for key, value in metrics.items():
                wandb.run.summary[f"final_{key}"] = value
            wandb.finish()

        results[env_name] = env_results

        print(f"\nAggregate results for {env_name}:")
        for key in metrics_keys:
            values = [r[key] for r in env_results]
            print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a distance metric")
    parser.add_argument("config", help="Path to a config file (see Runs/configs/)")
    args = parser.parse_args()

    mod = load_config(args.config)
    config, environments, seeds = mod.CONFIG, mod.ENVIRONMENTS, mod.SEEDS
    evaluate(seeds, environments, config)
