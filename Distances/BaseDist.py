"""
Base class shared by MAD-family distance metrics (MadDist, TDMadDist).
Provides common training, evaluation, and serialization logic.
"""

from collections import deque
from typing import Tuple, List, Any
import torch
from tqdm import trange, tqdm
from Distances.Distance import Distance


class BaseDist(Distance):
    """
    Shared base for MAD-family distance metrics.

    Subclasses must set `self.exp_rep`, `self.model`, `self.optimizer`, and `self.config`
    in their `__init__`, then this class provides all common functionality.
    """

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.config["device"])

    def add_trajectories(self, trajectories: List[List[Any]]) -> None:
        for trajectory in tqdm(trajectories, desc="Adding trajectories to the ER"):
            self.add_trajectory(trajectory)

    def train(self, steps: int = 1, verbose: bool = False) -> Tuple[float, float, float]:
        """
        Train the distance model for the given number of steps.

        Returns:
            Tuple of average (objective loss, constraint loss, total loss).
        """
        self.model.train()
        loss_o_mw = deque(maxlen=100)
        loss_c_mw = deque(maxlen=100)
        loss_mw = deque(maxlen=100)

        if verbose:
            t = trange(steps, desc='Steps of gradient', leave=True)
            for _ in t:
                loss_o, loss_c, loss = self.model.training_step(self.exp_rep, self.optimizer, self.config)
                loss_o_mw.append(loss_o.item())
                loss_c_mw.append(loss_c.item())
                loss_mw.append(loss.item())
                t.set_description("Steps of gradient, loss_o: %f, loss_c: %f, loss: %f" % (
                    sum(loss_o_mw) / len(loss_o_mw),
                    sum(loss_c_mw) / len(loss_c_mw),
                    sum(loss_mw) / len(loss_mw)
                ))
                t.refresh()
        else:
            for _ in range(steps):
                loss_o, loss_c, loss = self.model.training_step(self.exp_rep, self.optimizer, self.config)
                loss_o_mw.append(loss_o.item())
                loss_c_mw.append(loss_c.item())
                loss_mw.append(loss.item())

        return (
            float(sum(loss_o_mw) / len(loss_o_mw)),
            float(sum(loss_c_mw) / len(loss_c_mw)),
            float(sum(loss_mw) / len(loss_mw)),
        )

    def eval_dist(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model.dist(
                self.model.encoder(self.to_device(s1)),
                self.model.encoder(self.to_device(s2))
            )

    def eval_z_dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model.dist(self.to_device(z1), self.to_device(z2))

    def eval_embed_state(self, s: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model.encoder(self.to_device(s))

    def save(self, save_path: str) -> None:
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(state, save_path)
