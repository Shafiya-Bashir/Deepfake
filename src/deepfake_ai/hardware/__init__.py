"""Hardware detection, device management, and telemetry module."""

from deepfake_ai.hardware.hardware_detector import HardwareProfile, detect_hardware
from deepfake_ai.hardware.memory_monitor import MemoryMonitor
from deepfake_ai.hardware.device_manager import DeviceManager
from deepfake_ai.hardware.resource_limits import check_system_resources

__all__ = [
    'HardwareProfile',
    'detect_hardware',
    'MemoryMonitor',
    'DeviceManager',
    'check_system_resources',
]
