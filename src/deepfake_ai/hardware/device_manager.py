"""Device placement and execution management."""

import torch
from typing import Optional
from deepfake_ai.hardware.hardware_detector import detect_hardware


class DeviceManager:
    """Manages target PyTorch device selection."""

    def __init__(self, requested_device: Optional[str] = None):
        profile = detect_hardware()
        if requested_device is None or requested_device == 'auto':
            self.device_str = profile.recommended_device
        else:
            self.device_str = requested_device
        self.device = torch.device(self.device_str)

    def to_device(self, obj):
        """Moves tensor or module to active device."""
        if hasattr(obj, 'to'):
            return obj.to(self.device)
        return obj

    def __str__(self):
        return f'DeviceManager(device={self.device_str})'
