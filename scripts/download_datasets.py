#!/usr/bin/env python3
"""Dataset downloader and manifest generator."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake_ai.data.dataset_downloader import DatasetDownloader


def main():
    parser = argparse.ArgumentParser(description='DeepFake Dataset Downloader')
    parser.add_argument('--mode', choices=['sample', 'full'], default='sample', help='Download sample test fixtures or full dataset')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent / 'data'
    DatasetDownloader.setup_sample_data(base_dir)
    print(f'Dataset scaffolding complete in mode: {args.mode}')


if __name__ == '__main__':
    main()
