"""MAD distance evaluation config — antmaze environments (ICML 2026)."""
import torch
from Envs.CliffWalking import NormalizedCliffWalking
from Envs.GridWorlds.EmptyGridWorld import EmptyGridWorld
from Envs.GridWorlds.KeyDoorGridWorld import KeyDoorGridWorld
from Envs.Mediummaze import Mediummaze
from Envs.OgbenchAntmaze import OgbenchAntmaze, OgbenchAntmazeDeterministicTeleport
from Envs.OgbenchPointmaze import OgbenchPointmaze
from Envs.Umaze import Umaze

CONFIG = {
    "distance_class": "mad",
    "track": True,
    "debug": False,
    "exp_name": "MAD",
    "job_type": "icml2026",
    "project": "MAD_Dist",
    "gradient_steps": 200000,
    "er_max_n_traj": 1000,
    "prioritization": False,
    "out_d": 512,
    "hidden_dims": [512, 512, 256, 128],
    "d_type": "Simple",
    "dim_per_comp": 16,
    "in_dist_d": 32,
    "out_dist_d": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "l_rate": 0.0001,
    "batch_size_o": 256,
    "batch_size_c": 1024,
    "max_dist_obj": 5e2,
    "max_dist_accuracy": 5e2,
    "max_dist_con": 6,
    "weight_objective_1": 0.1,
    "weight_objective_2": 1.0,
    "weight_constrains": 0.001,
    "max_grad_norm": 5.0,
    "scaling_factor": 1,
    "amsgrad": False,
}

ENVIRONMENTS = [
    ("KeyDoor", lambda: KeyDoorGridWorld()),
    ("EmptyGridWorld", lambda: EmptyGridWorld()),
    ("CliffWalking", lambda: NormalizedCliffWalking()),
    ("Umaze", lambda: Umaze()),
    ("Mediummaze", lambda: Mediummaze()),
    ("ogbench-pm-medium-navigate", lambda: OgbenchPointmaze(env_name='pointmaze-medium-navigate-v0')),
    ("ogbench-pm-medium-stitch", lambda: OgbenchPointmaze(env_name='pointmaze-medium-stitch-v0')),
    ("ogbench-pm-large-navigate", lambda: OgbenchPointmaze(env_name='pointmaze-large-navigate-v0')),
    ("ogbench-pm-large-stitch", lambda: OgbenchPointmaze(env_name='pointmaze-large-stitch-v0')),
    ("ogbench-pm-giant-navigate", lambda: OgbenchPointmaze(env_name='pointmaze-giant-navigate-v0')),
    ("ogbench-pm-giant-stitch", lambda: OgbenchPointmaze(env_name='pointmaze-giant-stitch-v0')),
    ("antmaze-medium-navigate-v0", lambda: OgbenchAntmaze(env_name='antmaze-medium-navigate-v0')),
    ("antmaze-large-navigate-v0", lambda: OgbenchAntmaze(env_name='antmaze-large-navigate-v0')),
    ("antmaze-giant-navigate-v0", lambda: OgbenchAntmaze(env_name='antmaze-giant-navigate-v0')),
    ("antmaze-medium-stitch-v0", lambda: OgbenchAntmaze(env_name='antmaze-medium-stitch-v0')),
    ("antmaze-large-stitch-v0", lambda: OgbenchAntmaze(env_name='antmaze-large-stitch-v0')),
    ("antmaze-giant-stitch-v0", lambda: OgbenchAntmaze(env_name='antmaze-giant-stitch-v0')),
    ("antmaze-medium-explore-v0", lambda: OgbenchAntmaze(env_name='antmaze-medium-explore-v0')),
    ("antmaze-teleport-v0-det", lambda: OgbenchAntmazeDeterministicTeleport()),
]

SEEDS = [0, 1, 2, 3, 4]
