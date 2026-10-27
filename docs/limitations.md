# Limitations and Failure Modes

While DeepFake-AI achieves state-of-the-art results across standard benchmarks, practitioners should be aware of known failure modes:

## 1. Heavy Social Media Re-Compression
Aggressive compression codecs (e.g. WhatsApp video transcoding or TikTok delivery at $< 500 \text{ kbps}$) filter out high-frequency spatial components ($u, v > 32$ in 2D-DCT), lowering model confidence and shifting predictions toward neutral uncertainty ($P \approx 0.5$).

## 2. Unseen Next-Generation Diffusion Models
Flow-matching and latent consistency models (e.g., SD3, FLUX, Sora) generate continuous video with temporal coherence that surpasses older autoregressive or GAN synthesis. Zero-shot transfer to newly released generative engines experiences a temporary performance drop until fine-tuned on latent residual features.

## 3. Extreme Facial Occlusion & Profile Angles
Facial yaw angles exceeding $60^\circ$ or heavy occlusions (surgical masks, microphones, thick glasses) degrade facial landmark alignment quality, reducing the reliability of face-boundary forensic features.
