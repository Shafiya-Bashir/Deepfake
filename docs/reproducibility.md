# Reproducibility Guide & Experimental Controls

To guarantee that all experimental benchmarks, tables, and ROC curves can be independently replicated across different compute environments, DeepFake-AI implements strict determinism controls.

## 1. Global Random Seed Standardization
All pseudorandom number generators across Python, NumPy, PyTorch, and CUDA are pinned to `seed = 42`:
```python
import os, random, numpy as np, torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

## 2. Deterministic Dataset Splitting
Splits are generated strictly using the `actor_id` identity hash to prevent identity leakage:
$$\text{Split}(\text{Actor}) = \text{MD5}(\text{Actor ID} \,\|\, \text{Salt}) \pmod{100}$$
- `[0, 69]`: Training (70%)
- `[70, 84]`: Validation (15%)
- `[85, 99]`: Test (15%)

## 3. Checkpoint & Environment Manifests
Every training run emits a run metadata JSON capturing:
- Git commit hash
- Python version, PyTorch version, torchvision version
- Complete CLI flags and YAML parameters
- Hardware device type and system memory capacity
