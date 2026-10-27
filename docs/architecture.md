# System Architecture: DeepFake-AI Platform

## 1. High-Level System Architecture

DeepFake-AI is designed as an enterprise-grade, multi-domain forensic verification and deepfake detection platform. It ingests raw image and video streams, isolates facial regions, extracts multi-domain representations (spatial, frequency, spatio-temporal), and evaluates them through a hierarchical multi-tier model ensemble.

```mermaid
flowchart TD
    subgraph Ingestion ["Media Ingestion & Face Alignment Layer"]
        A1["Raw Video / Images (.mp4, .avi, .png, .jpg)"] --> B1["Video Frame Demuxing & Sampling"]
        A2["Benchmark Datasets (FF++, Celeb-DF, DFDC, GenImage)"] --> B1
        B1 --> B2["Face Detection & Landmark Localization (MTCNN)"]
        B2 --> B3["Affine Face Alignment & 224x224 Margin Cropping"]
        B3 --> B4["Identity-Disjoint Splitting (70/15/15 Train/Val/Test)"]
    end

    subgraph FeatureStream ["Multi-Domain Feature Extraction"]
        B3 --> F1["Spatial Stream (Normalized RGB Tensor)"]
        B3 --> F2["Frequency Stream (2D-DCT & High-Pass Spectral Filter)"]
        B3 --> F3["Temporal Stream (Inter-Frame Optical Flow & Blinking)"]
        B3 --> F4["Diffusion Inversion Stream (DIRE Residuals)"]
    end

    subgraph HardwareLayer ["Hardware Acceleration & Telemetry"]
        H1["Hardware Detector"] --> H2["Device Selection (Apple MPS / NVIDIA CUDA / CPU)"]
        H2 --> H3["Memory Monitor & Resource Bounds"]
    end

    subgraph ModelLayer ["Model Zoo & Hybrid Inference Engine"]
        M1["Classical Tier (LBP/GLCM Features + SVM & Random Forest)"]
        M2["Spatial CNN Tier (MesoInception4, XceptionNet, EfficientNet-B4)"]
        M3["Dual-Stream Spectral Tier (Spatial-Frequency Cross-Attention)"]
        M4["Spatio-Temporal Tier (ResNet-LSTM Video Dynamics)"]
        M5["Diffusion Forensic Tier (Latent Reconstruction Error)"]
        M6["Ensemble Tier (Soft Voting & Stacking Meta-Learner)"]
    end

    subgraph ValidationLayer ["Evaluation & Benchmark Suite"]
        V1["Threshold Tuning (ROC-AUC / Equal Error Rate Optimization)"]
        V2["Metrics Suite (AUC, ACC, EER, LogLoss, Brier Score)"]
        V3["Statistical Significance Testing (DeLong AUC, McNemar)"]
    end

    subgraph ExplainabilityLayer ["Explainable AI & Forensic Reporting"]
        E1["Grad-CAM & Grad-CAM++ Saliency Maps"]
        E2["Spatial Blending Boundary Detection"]
        E3["Spectral Energy Distribution Histograms"]
        E4["Automated Forensic Verdict Reports"]
    end

    F1 & F2 & F3 & F4 --> ModelLayer
    HardwareLayer -.-> ModelLayer
    ModelLayer --> ValidationLayer
    ModelLayer --> ExplainabilityLayer
```

---

## 2. Core Architectural Components

### 2.1 Ingestion & Face Preprocessing
- **Frame Demuxer**: Decodes video streams with configurable temporal stride ($k=2$ or $4$) to balance speed and temporal resolution.
- **Landmark-Based Affine Warping**: Aligns the eye centers horizontally to normalize pose variation.
- **Identity-Disjoint Guard**: Strict splitting ensuring no actor identities cross train/val/test partitions.

### 2.2 Forensic Multi-Domain Feature Extraction
- **Spatial RGB Tensor**: Normalized using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **2D-DCT Spectral Map**: Decomposes spatial image blocks into 64 frequency channels; high-frequency components are isolated via an octave high-pass mask.
- **Temporal Sequence Tensor**: Sequences of $T=16$ consecutive aligned face frames feeding recurrence and 3D convolutions.

### 2.3 Model Zoo Tiering
- **Tier 1 (Classical Forensics)**: Fast CPU inference utilizing LBP and GLCM texture statistics.
- **Tier 2 (Spatial Deep Learning)**: MesoNet and XceptionNet focusing on intra-frame boundary and blending seams.
- **Tier 3 (Dual-Stream Cross-Attention)**: Fuses spatial semantics with frequency fingerprints.
- **Tier 4 (Spatio-Temporal)**: Detects unnatural temporal jitter, eyelid freezing, and lip-sync anomalies.
- **Tier 5 (Ensemble)**: Calibrated weighted average delivering SOTA generalization across unseen generators.
