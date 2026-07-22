"""
This module implements the TD-MAD distance metric for reinforcement learning.
The TD-MAD distance extends the MAD distance by incorporating temporal difference learning
to better capture temporal relationships between states.
"""

from typing import List, Optional, Dict, Any
import torch
from Distances.BaseDist import BaseDist
from DistExpReplay.ErDist import ErDist
from Models.DistModels import TDMadDistEncoder


class TDMadDist(BaseDist):
    """
    Implementation of the Temporal Difference MAD (TD-MAD) distance metric.

    Extends BaseDist with a TD-learning objective and target network.

    Attributes:
        exp_rep (ErDist): Experience replay buffer
        model (TDMadDistEncoder): Neural network model with target encoder
        optimizer (torch.optim.AdamW): Optimizer for training
        config (dict): Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any], trajectories: Optional[List[List[Any]]] = None) -> None:
        """
        Initialize the TD-MAD distance model.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - er_max_n_traj (int): Maximum number of trajectories in experience replay
                - prioritization (str): Type of prioritization for experience replay
                - in_d (int): Input dimension
                - out_d (int): Output dimension
                - d_type (str): Distance type
                - in_dist_d (Optional[int]): Input distribution dimension
                - out_dist_d (Optional[int]): Output distribution dimension
                - dim_per_comp (Optional[int]): Dimensions per component
                - device (str): Device to run the model on
                - l_rate (float): Learning rate
                - amsgrad (bool): Whether to use AMSGrad variant of Adam
                - max_dist_con (int): Maximum distance constraint for trajectory addition
            trajectories (Optional[List[List[Any]]]): Initial trajectories to add to experience replay
        """
        self.exp_rep = ErDist(
            max_n_trajectories=config["er_max_n_traj"],
            trajectories_list=trajectories,
            prioritization=config["prioritization"],
            episode_len=100
        )

        self.model = TDMadDistEncoder(
            in_d=config["in_d"],
            out_d=config["out_d"],
            dist_type=config["d_type"],
            in_dist_d=config.get("in_dist_d", None),
            out_dist_d=config.get("out_dist_d", None),
            dim_per_component=config.get("dim_per_comp", None),
            hidden_dims=config.get("hidden_dims", None),
        ).to(config["device"])

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["l_rate"],
            eps=1e-10,
            weight_decay=0.,
            amsgrad=config.get("amsgrad", False)
        )
        self.config = config

    def add_trajectory(self, trajectory: List[Any]) -> None:
        """Add a single trajectory to the experience replay buffer."""
        self.exp_rep.add_trajectory(trajectory, max_d_c=self.config["max_dist_con"])

    @staticmethod
    def load(load_path: str) -> "TDMadDist":
        """
        Load a saved TDMadDist model.

        Args:
            load_path (str): Path to the saved model file.

        Returns:
            TDMadDist: Loaded instance with restored weights and optimizer state.
        """
        config = torch.load(load_path, map_location='cpu', weights_only=False)['config']
        device = config.get("device", "cpu")
        state = torch.load(load_path, map_location=device, weights_only=False)

        instance = TDMadDist(config=config)
        instance.model.load_state_dict(state['model_state'])
        instance.optimizer.load_state_dict(state['optimizer_state'])
        return instance
