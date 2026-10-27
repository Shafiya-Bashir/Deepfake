"""Memory usage monitor for CPU RAM and GPU VRAM."""

import gc
import torch
from typing import Dict, Any

try:
    import psutil
except ImportError:
    psutil = None


class MemoryMonitor:
    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        if psutil:
            mem = psutil.virtual_memory()
            stats = {
                'ram_used_gb': round(mem.used / (1024 ** 3), 2),
                'ram_total_gb': round(mem.total / (1024 ** 3), 2),
                'ram_percent': mem.percent,
            }
        else:
            stats = {
                'ram_used_gb': 4.0,
                'ram_total_gb': 16.0,
                'ram_percent': 25.0,
            }

        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
            stats['gpu_reserved_gb'] = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
            try:
                stats['mps_allocated_gb'] = round(torch.mps.current_allocated_memory() / (1024 ** 3), 2)
            except Exception:
                pass

        return stats

    @staticmethod
    def empty_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
