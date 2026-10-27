# Deep Learning Models for Deepfake Forensics

This directory contains deep neural network architectures and serialization checkpoints for media forgery detection:

## Model Categories
1. **Spatial Convolutional Networks**:
   - `Meso4` and `MesoInception4`: Microscopic mesoscale facial forgery analysis.
   - `XceptionNet`: Depthwise separable convolutional baseline, standard in FaceForensics++.
   - `EfficientNet-B4`: Scaled CNN with high capacity and compound scaling.
2. **Frequency Domain Networks**:
   - `DualStreamFrequencyNet`: Fuses spatial RGB representations with 2D Discrete Cosine Transform (DCT) high-pass feature maps via cross-attention.
3. **Spatio-Temporal Sequence Models**:
   - `SpatioTemporalDetector`: 3D-CNN / ResNet-LSTM modeling physiological motion cues, unnatural blink pauses, and lip-motion trajectories.
4. **Diffusion Residual Forensic Models**:
   - `DiffusionResidualDetector`: DIRE-based latent reconstruction error model targeting modern diffusion engines (Stable Diffusion, Midjourney).
