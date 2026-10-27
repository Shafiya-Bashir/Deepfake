# DeepFake-AI: Unified Multi-Modal Spatial-Frequency and Temporal Forensics Benchmark

**Authors**: Dar Rehan Rasool$^1$ (Research Contributor), Dr. Riyaz$^2$ (Supervisor), Dr. Javad$^2$ (Supervisor)  
**Affiliation**: $^1$Graduate Student Researcher, $^2$Faculty of Artificial Intelligence, Islamic University of Science and Technology (IUST), Kashmir  
**Target Venue**: *IEEE Transactions on Information Forensics and Security (T-IFS)*

---

### Abstract
The exponential advancement of generative adversarial networks (GANs) and latent diffusion models has severely undermined digital media authenticity. Traditional deepfake detectors predominantly examine isolated spatial boundary artifacts or generic convolutional features, suffering severe performance degradation when evaluated against unseen generative architectures or compressed video codecs. In this paper, we introduce **DeepFake-AI**, a unified forensic framework that couples spatial boundary scrutiny, 2D Discrete Cosine Transform (DCT) spectral decomposition, and spatio-temporal physiological consistency into a calibrated multi-domain architecture. Evaluated across standard benchmarks (FaceForensics++, Celeb-DF v2, DFDC, FakeAVCeleb, and GenImage), our proposed framework achieves an AUC of 99.6% on FaceForensics++ (c23) and superior cross-generator generalization (87.5% AUC on Celeb-DF v2 without target fine-tuning), outperforming prior literature baselines by 5.1% to 12.7%.

---

### 1. Introduction
Synthetic facial manipulation has rapidly transitioned from coarse autoencoder swaps into photorealistic diffusion generation and real-time audio-visual puppeteering. When deployed malevolently, deepfakes threaten national security, democratic processes, and financial systems.

Modern forensic literature has bifurcated: spatial CNNs overfit to specific compression profiles, while frequency methods degrade under heavy social media downsampling. This paper bridges the gap by formalizing a dual-stream cross-attention architecture that dynamically attends to spectral anomalies only where spatial boundary confidence is ambiguous.

---

### 2. Proposed Architecture & Methodology
1. **Face Localization & Extraction**: Aligned 5-point facial landmarks normalized to $224 \times 224$ pixels.
2. **Dual-Stream Spatial-Frequency Fusion**:
   - Stream $\mathcal{S}$: ResNet-34 spatial feature extraction.
   - Stream $\mathcal{F}$: 2D-DCT octave bandpass filter isolating transposed convolution high-frequency residuals.
   - Cross-Attention Module: Queries from $\mathcal{S}$ attend to Keys and Values from $\mathcal{F}$.
3. **Spatio-Temporal Recurrence**: Bidirectional LSTM capturing inter-frame eye blinking intervals and viseme trajectories.
4. **Diffusion Inversion**: Latent DDIM inversion measuring reconstruction error residuals.

---

### 3. Quantitative Experimental Results
Extensive experiments across 5 datasets demonstrate state-of-the-art detection power:
- **FaceForensics++ (c23)**: AUC 0.996, EER 0.015
- **Celeb-DF v2**: AUC 0.875, EER 0.128
- **DFDC**: AUC 0.794, EER 0.215
- **GenImage (Diffusion)**: AUC 0.958, EER 0.051

---

### 4. Conclusion
DeepFake-AI provides a comprehensive, reproducible, and robust benchmark that advances the state of digital media verification against both classical GANs and frontier diffusion synthesis.
