"""
This module implements the MAD distance metric for reinforcement learning.
"""

from typing import List, Optional, Dict, Any
import torch
from Distances.BaseDist import BaseDist
from DistExpReplay.ErDist import ErDist
from Models.DistModels import MadDistEncoder


class MadDist(BaseDist):
    """
    Implementation of the MAD Distance (MADdist) metric.
    This class learns a distance function between states in a latent space using a neural network.
    It maintains an experience replay buffer for training and provides methods for evaluating
    distances between states and their embeddings.

    Attributes:
        exp_rep (ErDist): Experience replay buffer for storing and sampling trajectories
        model (MadDistEncoder): Neural network model for encoding states and computing distances
        optimizer (torch.optim.AdamW): Optimizer for training the model
        config (dict): Configuration dictionary containing hyperparameters
    """

    def __init__(self, config: Dict[str, Any], trajectories: Optional[List[List[Any]]] = None) -> None:
        """
        Initialize the MAD distance model.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - er_max_n_traj (int): Maximum number of trajectories in experience replay
                - prioritization (str): Type of prioritization for experience replay
                - in_d (int): Input dimension
                - out_d (int): Output dimension
                - d_type (str): Distance type
                - in_dist_d (Optional[int]): Input WideNorm distance dimension
                - out_dist_d (Optional[int]): Output WideNorm distance dimension
                - dim_per_comp (Optional[int]): IQE dimensions per component
                - device (str): Device to run the model on
                - l_rate (float): Learning rate
                - amsgrad (bool): Whether to use AMSGrad variant of Adam
            trajectories (Optional[List[List[Any]]]): Initial trajectories to add to experience replay
        """
        self.exp_rep = ErDist(
            max_n_trajectories=config["er_max_n_traj"],
            trajectories_list=trajectories,
            prioritization=config["prioritization"],
            episode_len=100
        )

        self.model = MadDistEncoder(
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
        self.exp_rep.add_trajectory(trajectory, max_d_c=self.config.get("max_dist_con", None))

    @staticmethod
    def load(load_path: str) -> "MadDist":
        """
        Load a saved MadDist model.

        Args:
            load_path (str): Path to the saved model file.

        Returns:
            MadDist: Loaded instance with restored weights and optimizer state.
        """
        config = torch.load(load_path, map_location='cpu', weights_only=False)['config']
        device = config.get("device", "cpu")
        state = torch.load(load_path, map_location=device, weights_only=False)

        instance = MadDist(config=config)
        instance.model.load_state_dict(state['model_state'])
        instance.optimizer.load_state_dict(state['optimizer_state'])
        return instance
