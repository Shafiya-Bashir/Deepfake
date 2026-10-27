# Installation & Environment Setup Guide

## 1. System Requirements
- **Operating Systems**: macOS (Apple Silicon M1/M2/M3/M4 or Intel), Linux (Ubuntu 20.04/22.04 LTS), Windows 11 (WSL2).
- **Python**: Version 3.10, 3.11, or 3.12 recommended.
- **Hardware Acceleration**:
  - **Apple Silicon**: Native Metal Performance Shaders (`mps`).
  - **NVIDIA GPU**: CUDA 11.8+ or 12.1+ with cuDNN 8.8+.
  - **Memory**: Minimum 16 GB RAM (32 GB recommended for video decoding).

---

## 2. Installation Steps

### Step 1: Clone and Navigate
```bash
cd /Users/darrehanrasool/Desktop/Deep\ 
```

### Step 2: Virtual Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Core Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### Step 4: Verify Hardware Acceleration
Run the hardware diagnostic script to confirm device detection:
```bash
python3 scripts/verify_hardware.py
```
Expected output:
```
[INFO] DeepFake-AI Hardware Diagnostic
[INFO] PyTorch Version: 2.x
[INFO] Device Detected: MPS (Apple Silicon Metal) or CUDA (NVIDIA)
[INFO] System RAM Available: 16+ GB
```
