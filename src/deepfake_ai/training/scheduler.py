"""Learning rate scheduler factory."""

import torch
from typing import Dict, Any


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    name = config.get('name', 'CosineAnnealingWarmRestarts')
    if name == 'CosineAnnealingWarmRestarts':
        t_0 = int(config.get('T_0', 10))
        t_mult = int(config.get('T_mult', 2))
        eta_min = float(config.get('eta_min', 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
        )
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
