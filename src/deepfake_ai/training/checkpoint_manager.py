"""Checkpoint serialization and atomic storage."""

from pathlib import Path
import torch
from typing import Dict, Any


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_dir = self.checkpoint_dir / 'best'
        self.latest_dir = self.checkpoint_dir / 'latest'
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        model_name: str = 'model'
    ):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        latest_path = self.latest_dir / f'{model_name}_latest.pt'
        torch.save(state, latest_path)

        if is_best:
            best_path = self.best_dir / f'{model_name}_best.pt'
            torch.save(state, best_path)
