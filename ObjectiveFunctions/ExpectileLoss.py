import torch
import torch.nn as nn


class ExpectileLoss(nn.Module):
    def __init__(self, expectile: float):
        super().__init__()
        self.expectile = expectile

    def forward(self, advantage, diff):
        weight = torch.where(advantage >= 0, self.expectile, 1 - self.expectile)
        return (weight * (diff**2)).mean()
