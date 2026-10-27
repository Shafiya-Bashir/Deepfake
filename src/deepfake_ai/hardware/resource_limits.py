"""System resource assertions and safety limits."""

from deepfake_ai.core.exceptions import HardwareError

try:
    import psutil
except ImportError:
    psutil = None


def check_system_resources(min_ram_gb: float = 4.0):
    if psutil:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        if available_gb < min_ram_gb:
            raise HardwareError(
                f'Insufficient RAM: {available_gb:.2f} GB available, minimum {min_ram_gb} GB required.'
            )
