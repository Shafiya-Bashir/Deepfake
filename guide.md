# DeepFake-AI: User and Developer Guide

Welcome to **DeepFake-AI**, a state-of-the-art framework for deepfake detection, forensic artifact analysis, and multimodal media verification.

## 1. Quick Start Commands

```bash
# Setup environment and directories
make setup

# Download / verify datasets
python3 scripts/download_datasets.py --mode sample
python3 scripts/verify_datasets.py

# Prepare data (face detection, alignment, frequency transforms)
python3 scripts/prepare_data.py

# Train models
python3 scripts/train_all.py --config configs/training.yaml

# Run evaluation benchmarks
python3 scripts/evaluate_all.py

# Launch interactive forensics dashboard
python3 scripts/run_dashboard.py
```

## 2. Supported Datasets & Benchmarks
- **FaceForensics++ (FF++)**: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter (c23 / c40)
- **Celeb-DF v2**: High-quality face swapping with subtle boundaries
- **DFDC (Deepfake Detection Challenge)**: Diverse lighting, motion, and unconstrained environments
- **FakeAVCeleb**: Multimodal audio-visual manipulation and lipsync anomalies
- **GenImage**: Diffusion-generated faces (Stable Diffusion, Midjourney, DALL-E) vs GANs

## 3. Supported Model Architectures
- **Spatial Deep Models**: MesoNet (Meso4, MesoInception4), XceptionNet, EfficientNet-B4
- **Frequency Domain**: Dual-stream Spatial-Frequency Network (RGB + 2D-DCT high-pass residual)
- **Spatio-Temporal**: Spatio-temporal ResNet-LSTM / 3D-CNN for temporal flickering & unnatural blinking
- **Diffusion Forensics**: DIRE-inspired latent reconstruction error classifier
- **Ensemble**: Soft/Hard voting & Stacking ensemble
