"""Experiment run logging and metrics serialization."""

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any


class ExperimentLogger:
    """Logs experiment execution parameters and final evaluation results."""

    def __init__(self, registry_path: str = 'experiments/experiment_registry.csv'):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, experiment_data: Dict[str, Any]):
        file_exists = self.registry_path.exists()
        fieldnames = list(experiment_data.keys())

        with open(self.registry_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(experiment_data)
