"""
This module implements the original MAD distance metric from the paper
"State Representation Learning for Goal-Conditioned Reinforcement Learning"
(Steccanella & Jonsson, 2022), for comparison with the improved MadDist.
"""

import multiprocessing
from typing import List, Optional, Dict, Any
import torch
from Distances.BaseDist import BaseDist
from DistExpReplay.ErDist import ErDist
from Models.DistModels import MadDistOrEncoder

multiprocessing.set_start_method('spawn', force=True)


class MadDistOr(BaseDist):
    """
    Implementation of the original MAD distance metric from the paper (equation 4).

    Loss = (1/d_TD^2) * (||φ(s) - φ(s')||_1 - d_TD)^2
         + (1/d_TD^2) * max(0, ||φ(s) - φ(s')||_1 - d_TD)^2

    This is a simpler baseline compared to the improved MadDist, used to
    measure the benefit of the enhancements introduced in later work.

    Required config keys:
        er_max_n_traj (int):   Max trajectories in experience replay
        in_d (int):            Input state dimension
        out_d (int):           Latent space dimension
        d_type (str):          Distance type — "L1" (paper default) or "Simple" (asymmetric)
        device (str):          Torch device ("cpu" or "cuda")
        l_rate (float):        Learning rate
        batch_size_o (int):    Batch size for each training step
        max_dist_traj_batch (int): Distance threshold when sampling training pairs
        scaling_factor (float): Multiplier applied to trajectory distances
        max_grad_norm (float): Gradient clipping norm (None to disable)
    """

    def __init__(self, config: Dict[str, Any], trajectories: Optional[List[List[Any]]] = None) -> None:
        self.exp_rep = ErDist(
            max_n_trajectories=config["er_max_n_traj"],
            trajectories_list=trajectories,
            prioritization=False,
            episode_len=100,
        )

        self.model = MadDistOrEncoder(
            in_d=config["in_d"],
            out_d=config["out_d"],
            dist_type=config.get("d_type", "L1"),
        ).to(config["device"])

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["l_rate"],
            eps=1e-10,
            weight_decay=0.,
            amsgrad=config.get("amsgrad", False),
        )
        self.config = config

    def add_trajectory(self, trajectory: List[Any]) -> None:
        self.exp_rep.add_trajectory(trajectory, max_d_c=self.config.get("max_dist_con", None))

    @staticmethod
    def load(load_path: str) -> "MadDistOr":
        config = torch.load(load_path, map_location='cpu', weights_only=False)['config']
        device = config.get("device", "cpu")
        state = torch.load(load_path, map_location=device, weights_only=False)

        instance = MadDistOr(config=config)
        instance.model.load_state_dict(state['model_state'])
        instance.optimizer.load_state_dict(state['optimizer_state'])
        return instance
