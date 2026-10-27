"""Optimizer factory for AdamW, Adam, and SGD."""

import torch
import torch.nn as nn
from typing import Dict, Any


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    name = config.get('name', 'AdamW').lower()
    lr = float(config.get('lr', 0.0002))
    weight_decay = float(config.get('weight_decay', 0.01))

    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr)
