#!/usr/bin/env python3
"""Data preprocessing: face extraction, 2D-DCT frequency decomposition, and splits."""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake_ai.data.data_splitting import split_identity_disjoint


def main():
    base = Path(__file__).resolve().parent.parent / 'data'
    meta_path = base / 'processed/clean_metadata.csv'
    if meta_path.exists():
        df = pd.read_csv(meta_path)
        train_df, val_df, test_df = split_identity_disjoint(df)
        train_df.to_csv(base / 'splits/train/train_samples.csv', index=False)
        val_df.to_csv(base / 'splits/validation/val_samples.csv', index=False)
        test_df.to_csv(base / 'splits/test/test_samples.csv', index=False)
        print(f'Data prepared successfully! Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}')
    else:
        print('No clean metadata found to split.')


if __name__ == '__main__':
    main()
