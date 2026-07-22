from abc import ABC, abstractmethod
from typing import Union
import torch


class Distance(ABC):

    @abstractmethod
    def add_trajectory(self, trajectory: list):

        pass

    @abstractmethod
    def train(self, steps: int = 1):

        pass

    @abstractmethod
    def eval_dist(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:

        pass

    @abstractmethod
    def eval_z_dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        pass

    @abstractmethod
    def eval_embed_state(self, s: torch.Tensor) -> torch.Tensor:
        pass

    def save(self, save_path: str):
        pass

    @staticmethod
    def load(load_path: str):
        pass