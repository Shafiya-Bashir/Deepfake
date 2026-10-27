"""Hardware acceleration detection for CUDA, Apple MPS, and CPU."""

import os
import platform
import torch
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    cpu_architecture: str
    cpu_count_physical: int
    cpu_count_logical: int
    total_ram_gb: float
    available_ram_gb: float
    has_cuda: bool
    cuda_device_count: int
    cuda_device_name: Optional[str]
    has_mps: bool
    recommended_device: str


def detect_hardware() -> HardwareProfile:
    os_name = platform.system()
    os_version = platform.release()
    arch = platform.machine()
    
    if psutil:
        phys_cpus = psutil.cpu_count(logical=False) or 1
        log_cpus = psutil.cpu_count(logical=True) or 1
        mem = psutil.virtual_memory()
        total_ram = round(mem.total / (1024 ** 3), 2)
        avail_ram = round(mem.available / (1024 ** 3), 2)
    else:
        phys_cpus = os.cpu_count() or 1
        log_cpus = os.cpu_count() or 1
        total_ram = 16.0
        avail_ram = 12.0

    has_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if has_cuda else 0
    cuda_name = torch.cuda.get_device_name(0) if has_cuda and cuda_count > 0 else None

    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if has_cuda:
        recommended = 'cuda'
    elif has_mps:
        recommended = 'mps'
    else:
        recommended = 'cpu'

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        cpu_architecture=arch,
        cpu_count_physical=phys_cpus,
        cpu_count_logical=log_cpus,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        has_cuda=has_cuda,
        cuda_device_count=cuda_count,
        cuda_device_name=cuda_name,
        has_mps=has_mps,
        recommended_device=recommended,
    )
