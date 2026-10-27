"""Crash recovery and interrupted run restoration."""

from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


class RecoveryManager:
    @staticmethod
    def restore(checkpoint_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> int:
        if not checkpoint_path.exists():
            return 0
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f'Restored model weights from {checkpoint_path} (resuming at epoch {start_epoch})')
        return start_epoch
