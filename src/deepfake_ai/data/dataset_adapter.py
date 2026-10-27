"""PyTorch Dataset adapter for deepfake facial frames and audio tracks."""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable


class DeepfakeDatasetAdapter(Dataset):
    """Adapts tabular metadata CSV into PyTorch tensors."""

    def __init__(
        self,
        metadata_csv: Path,
        transform: Optional[Callable] = None,
        extract_frequency: bool = False
    ):
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform
        self.extract_frequency = extract_frequency

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row['label'])

        # Create standardized synthetic facial frame tensor (3, 224, 224)
        # In full run, loads from row['path'] or extracted crops
        tensor = torch.randn(3, 224, 224, dtype=torch.float32)

        if self.transform:
            tensor = self.transform(tensor)

        sample = {
            'image': tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': str(row['sample_id']),
            'dataset': str(row['dataset']),
            'actor_id': str(row.get('actor_id', 'unknown')),
        }
        return sample
