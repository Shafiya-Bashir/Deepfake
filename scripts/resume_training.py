#!/usr/bin/env python3
"""Resume training from an existing checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake_ai.training.recovery_manager import RecoveryManager
from deepfake_ai.models.model_factory import ModelFactory


def main():
    ckpt_path = Path(__file__).resolve().parent.parent / 'checkpoints/best/xception_best.pt'
    model = ModelFactory.create('xception')
    epoch = RecoveryManager.restore(ckpt_path, model)
    print(f'Ready to resume training from epoch {epoch}')


if __name__ == '__main__':
    main()
