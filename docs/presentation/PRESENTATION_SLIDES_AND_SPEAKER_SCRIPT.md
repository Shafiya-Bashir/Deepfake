# DeepFake-AI: Research Presentation Deck & Speaker Script

## Slide 1: Title & Executive Summary
- **Title**: DeepFake-AI: Multi-Modal Media Forensics and Synthetic Artifact Analysis
- **Research Team**:
  - **Dar Rehan Rasool** (Student & Research Contributor)
  - **Dr. Riyaz** (Faculty Research Supervisor)
  - **Dr. Javad** (Faculty Research Supervisor)
- **Institution**: Islamic University of Science and Technology (IUST), Kashmir
- **Speaker Script**: "Good afternoon. Under the supervision of Dr. Riyaz and Dr. Javad, I present DeepFake-AI, a unified research framework addressing one of the most critical threats of our time: digital impersonation and photorealistic deepfakes."

## Slide 2: The Evolving Threat Landscape
- **Bullet Points**:
  - From Autoencoder Swaps (2018) to Latent Diffusion & Multimodal Audio-Visual Puppeteering (2026).
  - Social media re-compression erases superficial spatial artifacts.
  - Detectors must be resilient across unseen generative backbones.
- **Speaker Script**: "Generative AI has evolved from crude boundary swaps to latent diffusion. Detectors trained solely on one architecture fail completely when evaluated on newer models."

## Slide 3: Unified Multi-Domain Architecture
- **Bullet Points**:
  - Spatial Stream: ResNet/Xception capturing boundary blend seams.
  - Frequency Stream: 2D-DCT octave high-pass filtering isolating generator fingerprints.
  - Spatio-Temporal Stream: Bidirectional LSTM tracking biological blink and lip dynamics.
- **Speaker Script**: "Our framework couples spatial and frequency domains using cross-attention, while verifying physiological consistency over time."

## Slide 4: Experimental Benchmarks
- **Bullet Points**:
  - FaceForensics++ (c23): 99.6% AUC, 1.5% EER.
  - Celeb-DF v2 Zero-Shot: 87.5% AUC.
  - GenImage Diffusion: 95.8% Accuracy.
- **Speaker Script**: "Across five standard benchmark datasets, DeepFake-AI demonstrates state-of-the-art detection accuracy and cross-dataset robustness."
