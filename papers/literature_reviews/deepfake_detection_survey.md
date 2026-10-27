# Comprehensive Survey of Deepfake Detection & Synthetic Media Forensics

## 1. Introduction & Problem Taxonomy
Deepfakes represent synthetic manipulations of human facial appearance, speech, and motion produced by deep generative models. The threat landscape has evolved across three major technological eras:
1. **Autoencoder & Face-Swap Era (2017-2019)**: DeepFaceLab, FaceSwap, NeuralTextures where encoder-decoder architectures blend donor facial expressions onto target identity frames.
2. **Generative Adversarial Network Era (2018-2022)**: ProGAN, StyleGAN, StyleGAN2/3 delivering photorealistic static images and First Order Motion Model (FOMM) reenactments.
3. **Latent Diffusion & Multimodal Era (2022-2026)**: Stable Diffusion, DALL-E 3, Midjourney, SadTalker, and diffusion-based voice cloning producing subtle high-frequency details without traditional GAN grid artifacts.

## 2. Detection Paradigms

### 2.1 Spatial Domain Forensics
- **Microscopic Artifacts (MesoNet)**: Early architectures revealed that compression and pooling operations in generative pipelines leave mesoscopic texture anomalies.
- **Boundary & Blending Inconsistencies (Face X-Ray)**: Most face-swapping pipelines paste a generated face onto a background frame, creating an intrinsic blending boundary where color gamut, resolution, and sensor noise exhibit step discontinuities.
- **Universal CNN Artifacts (Wang et al., CVPR 2020)**: Convolutional layers induce structural periodicities due to strided transpositions and interpolation kernels.

### 2.2 Frequency Domain Forensics
- **Spectral Checkerboard Artifacts (Frank et al., ICML 2020)**: Transposed convolutions (deconvolution) cause periodic checkerboard patterns in the 2D Discrete Cosine Transform (DCT) and Fourier Power Spectrum, manifesting as prominent frequency spikes.
- **Phase Spectrum Discontinuities**: While synthetic generators often match amplitude spectra of natural images, phase coherence across high-frequency bands is degraded.

### 2.3 Temporal & Physiological Dynamics
- **Spontaneous Eye Blinking (Li et al., WIFS 2018)**: Early GAN training sets primarily comprised static portraits with eyes open, causing synthetic subjects to exhibit abnormally low blink rates.
- **Visual Speech & Lip Dynamics (LipForensics, CVPR 2021)**: Learned high-level temporal representations of phoneme-viseme mappings expose semantic inconsistencies in audio-driven talking head models.

### 2.4 Diffusion-Specific Reconstruction Forensics
- **Diffusion Reconstruction Error (DIRE, ICCV 2023)**: When an image is inverted through DDIM into Gaussian noise and reconstructed, genuine camera images incur significant reconstruction error, whereas diffusion-generated images reconstruct with high fidelity.
- **Autoencoder Reconstruction Discrepancies (AEROBLADE, CVPR 2024)**: Exploits latent diffusion autoencoder reconstruction loss without requiring explicit model retraining.
