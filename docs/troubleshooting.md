# Troubleshooting & Common Issues

### 1. `MPS backend out of memory` on Apple Silicon
- **Cause**: PyTorch Metal Performance Shaders allocation exceeded unified memory threshold.
- **Solution**: Set the environment variable before launching Python:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
  ```
  Or reduce `batch_size` in `configs/training.yaml` to 16.

### 2. `cv2.error: (-215:Assertion failed) !ssize.empty()` during video reading
- **Cause**: Corrupt frame decoding or unsupported video codec.
- **Solution**: Run `python3 scripts/verify_datasets.py` to identify and isolate unreadable video files.

### 3. MTCNN fails to detect faces in profile views
- **Cause**: Severe head rotation beyond $45^\circ$.
- **Solution**: Enable fallback to OpenCV Haar Cascade or RetinaFace in `configs/data.yaml` under `preprocessing.face_detection.detector`.
