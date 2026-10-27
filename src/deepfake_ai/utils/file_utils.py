"""Safe file system operations and atomic JSON storage."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> Path:
    """Creates directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write_json(data: Any, path: Path, indent: int = 2):
    """Safely writes JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=indent)
    tmp_path.replace(path)


def safe_read_json(path: Path) -> Dict[str, Any]:
    """Reads JSON file safely."""
    with open(path, 'r') as f:
        return json.load(f)
