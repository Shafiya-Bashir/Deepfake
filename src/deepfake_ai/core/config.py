"""Configuration loading and validation module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import logging

from deepfake_ai.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Master configuration container for DeepFake-AI."""
    project: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split('.')
        current = self.__dict__
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
            if current is None:
                return default
        return current


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Loads a YAML file safely."""
    if not file_path.exists():
        logger.warning(f'Config file not found: {file_path}')
        return {}
    with open(file_path, 'r') as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f'Failed to parse YAML file {file_path}: {e}')


def load_config(config_dir: Optional[Path] = None) -> Config:
    """Loads all configuration files from the config directory."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent.parent.parent / 'configs'

    return Config(
        project=load_yaml(config_dir / 'project.yaml'),
        data=load_yaml(config_dir / 'data.yaml'),
        models=load_yaml(config_dir / 'models.yaml'),
        training=load_yaml(config_dir / 'training.yaml'),
        hardware=load_yaml(config_dir / 'hardware.yaml'),
        visualization=load_yaml(config_dir / 'visualization.yaml'),
    )
