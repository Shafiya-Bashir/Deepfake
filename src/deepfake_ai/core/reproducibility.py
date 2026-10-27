"""Seed fixing and deterministic execution controls."""

import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seeds Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    logger.info(f'Global random seed set to {seed} (deterministic={deterministic})')
    return seed
