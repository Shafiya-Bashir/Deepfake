#!/usr/bin/env python3
"""Environment setup and directory hierarchy initializer."""

import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / 'src'))

from deepfake_ai.core.reproducibility import seed_everything
from deepfake_ai.hardware.hardware_detector import detect_hardware


def main():
    print('Setting up DeepFake-AI research environment...')
    seed_everything(42)
    profile = detect_hardware()
    print(f'Hardware audit complete: detected {profile.recommended_device} on {profile.os_name}')
    print('Environment setup successfully!')


if __name__ == '__main__':
    main()
