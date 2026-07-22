"""MAD planning evaluation on OGBench with random-sampling agent."""
CONFIG = {
    "distance_class": "qrl",
    "planning_mode": "ogbench_sampling",
    "render_mode": None,
    "action_bins": 600,
    "lookahead": 50,
    "num_samples": 400,
    "smoothing_window": 1,
    "action_repeat": 10,
    "num_eval_episodes": 50,
    "num_cores": 5,
    "exp_name": "QRL_Planning",
    "job_type": "planning",
    "project": "MAD_Dist",
    "track": True,
}

RUNS = [
    {
        "env_name": "pointmaze-medium-navigate-v0",
        "model_name": "QRL_ogbench-pm-medium-navigate",
        "overrides": {"lookahead": 50, "action_repeat": 5},
    },
    {
        "env_name": "pointmaze-large-navigate-v0",
        "model_name": "QRL_ogbench-pm-medium-navigate",
        "overrides": {"lookahead": 100, "action_repeat": 10},
    },
    {
        "env_name": "pointmaze-giant-navigate-v0",
        "model_name": "QRL_ogbench-pm-medium-navigate",
        "overrides": {"lookahead": 300, "action_repeat": 40},
    },
]

SEEDS = [0, 1, 2, 3, 4]
