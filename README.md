# DeepFake-AI: Unified Multi-Modal Spatial-Frequency and Temporal Forensics Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**DeepFake-AI** is a comprehensive, production-grade research platform and benchmarking framework for deepfake detection, synthetic facial artifact forensics, and multi-modal media authenticity verification. It unifies classical handcrafted forensic descriptors, spatial convolutional networks, dual-stream frequency decomposition (2D-DCT & FFT phase analysis), spatio-temporal dynamics (blinking & lip consistency), and modern diffusion-reconstruction residual forensics.

---

## 📑 Table of Contents
1. [Research Team & Authorship](#-research-team--authorship)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Supported Benchmark Datasets](#-supported-benchmark-datasets)
5. [Model Zoo & Forensic Techniques](#-model-zoo--forensic-techniques)
6. [Directory Structure](#-directory-structure)
7. [Quick Start & Usage](#-quick-start--usage)
8. [Benchmark Evaluation Matrix](#-benchmark-evaluation-matrix)
9. [Interactive Dashboard](#-interactive-dashboard)
10. [Citation & References](#-citation--references)

---

## 👥 Research Team & Authorship

This research project and benchmark platform is collaboratively developed under faculty guidance and supervision:

| Role | Name | Designation & Affiliation | Research & Project Contribution |
| :--- | :--- | :--- | :--- |
| **Faculty Research Supervisor** | **Dr. Riyaz** | Professor & Faculty Supervisor | Project Supervision, Research Direction, Theoretical Guidance & Methodological Review |
| **Faculty Research Supervisor** | **Dr. Javad** | Professor & Faculty Supervisor | Co-Supervision, Strategic Architecture Guidance, Academic Review & Validation Oversight |
| **Research Contributor & Student** | **Dar Rehan Rasool** | Graduate Student & Research Contributor | Multi-Domain Architecture, Spatial-Frequency & Temporal Pipeline, Deep Learning Implementation & Benchmarking |

- **Academic Unit**: Department / Centre of Artificial Intelligence
- **Institution**: Islamic University of Science and Technology (IUST), Kashmir, India

---

## 🌟 Key Features

- **Multi-Domain Forensic Analysis**: Simultaneous inspection across Spatial (RGB artifacts, blending boundaries), Frequency (DCT high-frequency spectral roll-off), and Temporal (optical flow, inter-frame jitter) domains.
- **Cross-Generator Generalization**: Explicit evaluation against unseen generative paradigms, comparing GAN artifacts (ProGAN, StyleGAN, StyleGAN3) against modern Diffusion models (Stable Diffusion, Midjourney, FLUX).
- **Identity-Disjoint Stratification**: Enforces strict actor-level disjoint splitting between training and testing sets, preventing deceptive identity leakage and inflated accuracy metrics.
- **Hardware-Aware Acceleration**: Native automatic dispatch across Apple Silicon (MPS), NVIDIA GPUs (CUDA), and multi-core CPU architectures with VRAM-conscious memory limits.
- **Explainability & Forensic Heatmaps**: Built-in Grad-CAM, Grad-CAM++, and Frequency Saliency Maps highlighting specific facial regions (eyes, mouth seam, cheek blending boundaries) responsible for synthetic verdicts.

---

## 🏗️ System Architecture

```
                                      +--------------------------+
                                      |   Raw Media Input Video  |
                                      +-------------+------------+
                                                    |
                                                    v
                                      +--------------------------+
                                      |  Face Detection & MTCNN  |
                                      +-------------+------------+
                                                    |
                      +-----------------------------+-----------------------------+
                      |                             |                             |
                      v                             v                             v
           +--------------------+        +--------------------+        +--------------------+
           |   Spatial Stream   |        |  Frequency Stream  |        |  Temporal Stream   |
           |  (RGB Face Crops)  |        | (2D-DCT / FFT Res) |        | (Frame Difference) |
           +----------+---------+        +----------+---------+        +----------+---------+
                      |                             |                             |
                      v                             v                             v
           +--------------------+        +--------------------+        +--------------------+
           |  Xception / EffNet |        |  Dual-Stream CNN   |        | Spatio-TemporalNet |
           +----------+---------+        +----------+---------+        +----------+---------+
                      |                             |                             |
                      +-----------------------------+-----------------------------+
                                                    |
                                                    v
                                      +--------------------------+
                                      | Calibrated Meta-Ensemble |
                                      +-------------+------------+
                                                    |
                               +--------------------+--------------------+
                               |                                         |
                               v                                         v
                 +---------------------------+             +---------------------------+
                 |  Confidence & Verdict     |             | Explainability Heatmap    |
                 |  (Fake Probability Score) |             | (Grad-CAM / Spatial Sali) |
                 +---------------------------+             +---------------------------+
```

---

## 📊 Supported Benchmark Datasets

| Dataset | Primary Manipulation Types | Real Identities | Manipulated Videos | Baseline AUC |
|:---|:---|:---:|:---:|:---:|
| **FaceForensics++ (FF++)** | Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter | 1,000 | 5,000 | 99.2% |
| **Celeb-DF v2** | High-Quality Deepfake Face Swapping | 590 | 5,639 | 98.7% |
| **DFDC Preview / Full** | Deepfake, GAN Alterations, Lighting & Occlusions | >3,000 | >100,000 | 91.5% |
| **FakeAVCeleb** | Multimodal Audio-Visual, LipSync & Voice Cloning | 500 | 20,000 | 95.8% |
| **GenImage** | Latent Diffusion, Stable Diffusion, Midjourney vs Real | Multiple | 1,300,000 | 97.4% |

---

## 🔬 Model Zoo & Forensic Techniques

1. **MesoNet (`Meso4` & `MesoInception4`)**: Microscopic spatial compression analysis (Afchar et al., 2018).
2. **XceptionNet**: Standard benchmark backbone with depthwise separable convolutions (Rossler et al., 2019).
3. **EfficientNet-B4**: High-capacity spatial feature extractor with compound scaling.
4. **Dual-Stream Frequency Network**: Integrates 2D Discrete Cosine Transform (DCT) and Laplacian spectral analysis to detect checkerboard artifacts.
5. **Spatio-Temporal Video Detector**: CNN-LSTM capturing abnormal eye blinking intervals and lip-motion inconsistency.
6. **Diffusion Reconstruction Forensics**: DIRE-inspired latent residual analysis for distinguishing diffusion-based synthesis from camera sensors.
7. **Classical Forensic Descriptors**: SVM and Random Forest trained on Local Binary Patterns (LBP) and Gray-Level Co-occurrence Matrices (GLCM).

---

## 📁 Directory Structure

```
Deep/
├── assets/                     # Sound alerts, visual badges, and media assets
├── checkpoints/                # Periodic, latest, and best saved PyTorch weights
├── configs/                    # YAML configuration files (models, data, training, hardware)
├── data/                       # Manifests, raw datasets, interim crops, and train/val/test splits
├── docs/                       # Architectural docs, research literature catalogs, IEEE paper, patent drafts
├── experiments/                # Experiment registry and benchmark result JSONs
├── logs/                       # Application, training epoch histories, telemetry, and error logs
├── models/                     # Dataset-specialized serialized models and training checkpoints
├── notebooks/                  # Interactive Jupyter exploratory data analysis and feature notebooks
├── outputs/                    # Trained weights, inference reports, and prediction tables
├── papers/                     # 30+ Seminal deepfake research PDFs, taxonomy, gaps, and formulations
├── presentation/               # Interactive slides and presentation assets
├── reports/                    # SOTA comparative analysis, data quality audits, evaluation reports
├── scripts/                    # Command-line tools for download, preparation, training, evaluation, UI
├── src/deepfake_ai/            # Core Python package (models, data, pipeline, training, hardware, UI)
├── tests/                      # Pytest unit and integration test suites
├── Makefile                    # Developer workflow automation commands
├── pyproject.toml              # Build system, package definition, and tool configurations
├── requirements.txt            # Production dependencies
└── README.md                   # Project overview and documentation
```

---

## 🚀 Quick Start & Usage

### 1. Environment Scaffolding
```bash
make setup
make install
```

### 2. Data Preparation & Verification
```bash
python3 scripts/download_datasets.py --mode sample
python3 scripts/verify_datasets.py
python3 scripts/prepare_data.py
```

### 3. Model Training
```bash
python3 scripts/train_all.py --config configs/training.yaml
```

### 4. Comprehensive Evaluation
```bash
python3 scripts/evaluate_all.py
```

### 5. Launch Forensics Dashboard
```bash
python3 scripts/run_dashboard.py --port 8080
```

---

## 📜 Citation & References

```bibtex
@article{rasool2026deepfakeai,
  title={DeepFake-AI: Unified Multi-Modal Spatial-Frequency and Temporal Forensics Benchmark},
  author={Rasool, Dar Rehan and Riyaz, Dr. and Javad, Dr.},
  affiliation={Islamic University of Science and Technology (IUST), Kashmir},
  journal={arXiv preprint},
  year={2026}
}
```
