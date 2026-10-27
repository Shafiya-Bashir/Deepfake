# State-of-the-Art Deepfake Taxonomy: Generation vs. Detection

## 1. Generative Taxonomy
```
Synthetic Media
├── Facial Replacement (Face Swapping)
│   ├── Autoencoder-based (DeepFaceLab, FaceSwap)
│   └── 3D Pose / Mesh Blending (FaceShifter)
├── Facial Reenactment (Puppeteering)
│   ├── Expression Transfer (Face2Face)
│   └── Audio-Driven Talking Heads (SadTalker, Wav2Lip)
├── Attribute & Identity Synthesis
│   ├── Latent Space Manipulation (StyleGAN, StyleGAN3)
│   └── Text-to-Image Diffusion (Stable Diffusion, Midjourney, DALL-E)
└── Video Diffusion Synthesis (Sora, RunWay Gen-2, Pika)
```

## 2. Forensic Detection Taxonomy
```
Deepfake Detection
├── Spatial Domain Forensics
│   ├── Mesoscopic Feature Extraction (MesoNet)
│   ├── Blending Seam Analysis (Face X-Ray)
│   └── Universal CNN Fingerprints (Wang et al.)
├── Spectral & Frequency Forensics
│   ├── 2D-DCT High-Frequency Anomaly Detection (Frank et al.)
│   ├── Fourier Phase Discontinuity
│   └── Implicit Continuous Frequency Decomposition (Spectra 2026)
├── Temporal & Biological Dynamics
│   ├── Spontaneous Eye Blink Cadence (Li et al.)
│   ├── Spatio-Temporal Lip Motion (LipForensics)
│   └── Blood Volume Pulse (rPPG Photoplethysmography)
└── Reconstruction Inversion Forensics
    ├── Latent Inversion Error (DIRE 2023)
    └── Autoencoder Residual Discrepancy (AEROBLADE 2024)
```
