"""Soft and hard voting ensemble across deep learning backbones."""

import torch
import torch.nn as nn
from typing import List


class VotingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = [w / sum(weights) for w in weights]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = [torch.softmax(m(x), dim=1) * w for m, w in zip(self.models, self.weights)]
        avg_prob = torch.stack(probs, dim=0).sum(dim=0)
        return avg_prob
