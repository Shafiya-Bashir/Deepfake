"""Manifest parser and serialization for deepfake datasets."""

from pathlib import Path
from typing import Dict, Any
from deepfake_ai.utils.file_utils import safe_read_json, safe_write_json


def load_manifest(path: Path) -> Dict[str, Any]:
    return safe_read_json(path)


def save_manifest(manifest_data: Dict[str, Any], path: Path):
    safe_write_json(manifest_data, path)
