"""Continuous background system metrics telemetry."""

import time
from typing import Dict, Any
from deepfake_ai.hardware.memory_monitor import MemoryMonitor

try:
    import psutil
except ImportError:
    psutil = None


class SystemMonitor:
    @staticmethod
    def sample() -> Dict[str, Any]:
        mem = MemoryMonitor.get_memory_stats()
        cpu_pct = psutil.cpu_percent(interval=None) if psutil else 15.0
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_pct,
            **mem
        }
