"""Validation assertions for tensors and inputs."""

from pathlib import Path
import torch
from deepfake_ai.core.exceptions import DataProcessingError


def validate_file_exists(path: Path):
    if not path.exists():
        raise DataProcessingError(f'Expected file does not exist: {path}')


def validate_tensor_shape(tensor: torch.Tensor, expected_dims: int, name: str = 'tensor'):
    if tensor.dim() != expected_dims:
        raise DataProcessingError(
            f'Expected {name} to have {expected_dims} dimensions, got shape {tensor.shape}'
        )
