"""Dataset schema and path integrity validation."""

import pandas as pd
from deepfake_ai.core.exceptions import DataProcessingError


def validate_dataset_integrity(df: pd.DataFrame):
    required = ['sample_id', 'label', 'dataset']
    for col in required:
        if col not in df.columns:
            raise DataProcessingError(f'Missing mandatory column: {col}')
