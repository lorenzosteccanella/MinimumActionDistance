"""HILBERT distance evaluation config."""
import torch
from Envs.CliffWalking import NormalizedCliffWalking
from Envs.GridWorlds.EmptyGridWorld import EmptyGridWorld
from Envs.GridWorlds.KeyDoorGridWorld import KeyDoorGridWorld
from Envs.Mediummaze import Mediummaze
from Envs.OgbenchAntmaze import OgbenchAntmaze, OgbenchAntmazeDeterministicTeleport
from Envs.OgbenchPointmaze import OgbenchPointmaze
from Envs.Umaze import Umaze

CONFIG = {
    "distance_class": "hilbert",
    "track": True,
    "debug": False,
    "exp_name": "HILBERT",
    "job_type": "icml2026",
    "project": "MAD_Dist",
    "gradient_steps": 20000,
    "max_dist_accuracy": 100,
    "er_max_n_traj": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "obs_size": None,     # obs_size is set automatically at runtime
    "embedding_dim": 32,
    "hidden_dims": [512],
    "her_num_goals": 128,
    "alpha": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "expectile": 0.9,
    "batch_size": 1024,
    "buffer_size": 20_000_000,
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
