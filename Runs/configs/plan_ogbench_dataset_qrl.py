"""QRL planning evaluation on OGBench with dataset-guided agent.

QRL models are loaded from trained_models/{model_name}_seed_{seed}_traced.pt
via torch.jit.load() and wrapped in TorchScriptDistanceWrapper.
"""
CONFIG = {
    "distance_class": "qrl",
    "planning_mode": "ogbench_dataset_guided",
    "render_mode": None,
    "lookahead": 50,
    "num_samples": 50,
    "num_iterations": 1,
    "num_elites": 1,
    "k_neighbors": 50,
    "step_penalty": 0.01,
    "cost_metric": "last_state",
    "num_eval_episodes": 50,
    "max_episode_steps": 1000,
    "num_cores": 5,
    "exp_name": "QRL_Planning_DatasetGuided",
    "job_type": "planning",
    "project": "MAD_Dist",
    "track": True,
    # Stuck detection
    "stuck_window": 50,
    "stuck_threshold": 1.0,
    "stuck_goal_progress": 0.5,
    "stuck_patience": 3,
}

RUNS = [
    {
        "env_name": "antmaze-medium-navigate-v0",
        "model_name": "QRL_antmaze-medium-explore-v0",
    },
]

SEEDS = [0, 1, 2]
