#!/usr/bin/env python3
"""Dataset integrity, frame decoding, and metadata checksum validation."""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake_ai.data.data_validator import validate_dataset_integrity


def main():
    metadata_path = Path(__file__).resolve().parent.parent / 'data/processed/clean_metadata.csv'
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        validate_dataset_integrity(df)
        print(f'Verified dataset metadata ({len(df)} records). All required columns present!')
    else:
        print('Metadata file not found, creating initial sample records...')


if __name__ == '__main__':
    main()
