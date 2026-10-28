#!/usr/bin/env python3
"""
generate_publication_diagrams.py
=================================
Generates 4 publication-grade, research-caliber architectural diagrams:
1. audio_architecture.png: Acoustic Waveform & Spectral Flatness Forensics
2. video_architecture.png: Spatio-Temporal Viseme & Lip-Sync Forensics
3. image_architecture.png: Multi-Scale Spatial-Frequency 2D-DCT Forensics
4. deepfake_ai_master_architecture.png: Master Multi-Modal System Architecture

Author: Dar Rehan Rasool (@darrehanrasool)
Supervisors: Dr. Riyaz* & Dr. Javad* (IUST)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "assets" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling Palette (Modern Cyber-Forensic Dark Mode)
BG_COLOR = "#070b14"
CARD_BG = "#0f172a"
CARD_BORDER = "#334155"
ACCENT_BLUE = "#2563eb"
ACCENT_CYAN = "#06b6d4"
ACCENT_SKY = "#38bdf8"
ACCENT_PURPLE = "#a855f7"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"


def draw_box(ax, x, y, w, h, title, subtitle="", box_color=CARD_BG, border_color=CARD_BORDER, title_color=TEXT_MAIN, title_size=10, sub_size=8, pad=0.02):
    """Draws a rounded rectangular card with guaranteed non-overflowing text."""
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size=0.03",
        facecolor=box_color,
        edgecolor=border_color,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(fancy)

    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color=title_color, fontsize=title_size, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", color=TEXT_MUTED, fontsize=sub_size, zorder=3, linespacing=1.2)
    else:
        ax.text(x + w / 2, y + h * 0.5, title, ha="center", va="center", color=title_color, fontsize=title_size, fontweight="bold", zorder=3, linespacing=1.2)


def draw_arrow(ax, x1, y1, x2, y2, color=ACCENT_SKY, width=1.5, text=""):
    """Draws an orthogonal or direct arrow with optional centered text label."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=width,
            mutation_scale=12,
            shrinkA=4,
            shrinkB=4
        ),
        zorder=1
    )
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.02
        ax.text(mx, my, text, ha="center", va="bottom", color=TEXT_MUTED, fontsize=7.5, fontweight="bold", zorder=4)


# ==============================================================================
# 1. AUDIO ARCHITECTURE DIAGRAM
# ==============================================================================
def generate_audio_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 0.75)
    ax.axis("off")

    # Header
    ax.text(0.7, 0.71, "Acoustic Speech Forensics & Synthetic Voice Clone Pipeline", ha="center", va="center", color=TEXT_MAIN, fontsize=14, fontweight="heavy")
    ax.text(0.7, 0.67, "DeepFake-AI™ Acoustic Architecture • SFM Vocoder & Phase Dispersion Analysis", ha="center", va="center", color=ACCENT_CYAN, fontsize=9.5, fontweight="bold")

    # Stage 1: Input Stream
    draw_box(ax, 0.05, 0.28, 0.18, 0.24, "Input Acoustic Stream", "Live Mic / Audio (.wav/.mp3)\nPCM 16 kHz, Mono\nFraming: 25ms window, 10ms hop", box_color="#0b1329", border_color=ACCENT_CYAN, title_color=ACCENT_CYAN)

    # Stage 2: Dual Feature Extraction Streams (Top & Bottom)
    # Branch A: Short-Time Fourier Transform
    draw_box(ax, 0.33, 0.44, 0.28, 0.16, "Branch A: STFT Spectrum", "X(m, ω) = Σ x[n] w[n-m] e^(-jωn)\nActive-Bin Spectral Flatness (SFM)\nγ = exp(Mean(ln P)) / Mean(P)", box_color="#111c38", border_color=ACCENT_SKY)
    # Branch B: High-Frequency Vocoder Energy
    draw_box(ax, 0.33, 0.26, 0.28, 0.15, "Branch B: Vocoder Energy", "Cutoff fc = 3.0 kHz\nHigh-Band Energy Ratio (HF_ratio)\nSpectral Centroid (Brightness)", box_color="#111c38", border_color=ACCENT_PURPLE)
    # Branch C: LPCC & Phase Discrepancy
    draw_box(ax, 0.33, 0.08, 0.28, 0.15, "Branch C: LPCC Residuals", "Linear Prediction Cepstral Coeffs\nVocal Tract Filter Inverse Error\nPhase Dispersion Metric", box_color="#111c38", border_color=ACCENT_AMBER)

    # Connections from Input to Branches
    draw_arrow(ax, 0.23, 0.42, 0.33, 0.52, color=ACCENT_CYAN)
    draw_arrow(ax, 0.23, 0.40, 0.33, 0.33, color=ACCENT_CYAN)
    draw_arrow(ax, 0.23, 0.38, 0.33, 0.15, color=ACCENT_CYAN)

    # Stage 3: Statistical Feature Fusion & Normalization
    draw_box(ax, 0.71, 0.24, 0.26, 0.28, "Acoustic Discrepancy Fusion", "S_audio = Clip(7.5·HF + 2.2·SFM)\nS_spectral = Centroid / 2200 Hz\nNeural Vocoder Classifier\n(Tacotron, ElevenLabs, SV2TTS)", box_color="#0b1736", border_color=ACCENT_BLUE)

    # Connections to Fusion
    draw_arrow(ax, 0.61, 0.52, 0.71, 0.42, color=ACCENT_SKY)
    draw_arrow(ax, 0.61, 0.33, 0.71, 0.38, color=ACCENT_PURPLE)
    draw_arrow(ax, 0.61, 0.15, 0.71, 0.30, color=ACCENT_AMBER)

    # Stage 4: Calibration & Decision Output
    draw_box(ax, 1.07, 0.26, 0.26, 0.24, "Acoustic Verdict & XAI", "P(Fake) = σ(0.70·S_audio + 0.30·S_spec)\nOutput: REAL / CLONED VOICE\nDaubert-Compliant Metric Card\nAlert Beacon & Audio Signal", box_color="#062e26", border_color=ACCENT_GREEN, title_color="#34d399")

    draw_arrow(ax, 0.97, 0.38, 1.07, 0.38, color=ACCENT_GREEN, text="z = logit")

    # Footer Attribution
    ax.text(0.7, 0.02, "© 2026 Dar Rehan Rasool (@darrehanrasool) • Supervisors: Dr. Riyaz* & Dr. Javad* (IUST) • DeepFake-AI", ha="center", va="center", color=TEXT_MUTED, fontsize=8, fontfamily="monospace")

    out_path = OUTPUT_DIR / "audio_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"[✓] Generated: {out_path}")


# ==============================================================================
# 2. VIDEO ARCHITECTURE DIAGRAM
# ==============================================================================
def generate_video_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 0.75)
    ax.axis("off")

    # Header
    ax.text(0.7, 0.71, "ST2-SyncNet Spatio-Temporal Video Forensics Architecture", ha="center", va="center", color=TEXT_MAIN, fontsize=14, fontweight="heavy")
    ax.text(0.7, 0.67, "Multi-Modal Audio-Visual Lip-Sync, Eye Blinking Dynamics & 3D-CNN Discrepancy", ha="center", va="center", color=ACCENT_CYAN, fontsize=9.5, fontweight="bold")

    # Stage 1: Input Stream
    draw_box(ax, 0.05, 0.28, 0.18, 0.24, "Input Video Stream", "Live Video / MP4 / WebM\nFrame Demuxing (25/30 fps)\nAudio Track Extraction\nMTCNN Face Crop (224x224)", box_color="#0b1329", border_color=ACCENT_CYAN, title_color=ACCENT_CYAN)

    # Stage 2: Dual Stream Processing
    # Stream 1: Spatial Facial Frame
    draw_box(ax, 0.33, 0.46, 0.28, 0.16, "Stream 1: Spatial Viseme", "RGB Tensor (3 × 224 × 224)\nLip Bounding Box Extraction\nFacial Landmark Trajectory\nVisual Phoneme Mapping", box_color="#111c38", border_color=ACCENT_SKY)
    # Stream 2: Temporal 3D-CNN Sequence
    draw_box(ax, 0.33, 0.26, 0.28, 0.16, "Stream 2: Temporal 3D-CNN", "8-Frame Sequence (8 × 3 × 224 × 224)\n3D-ResNet Motion Extractor\nInter-Frame Flicker Score (S_temp)\nOptical Flow Continuity", box_color="#111c38", border_color=ACCENT_PURPLE)
    # Stream 3: Eye Aspect Ratio (EAR)
    draw_box(ax, 0.33, 0.06, 0.28, 0.16, "Stream 3: Biological EAR", "EAR = (||p2-p6|| + ||p3-p5||)/(2||p1-p4||)\nBlink Dynamics Derivative dE/dt\nPhysiological Regularity Score\nUncanny Valley Artifact Metric", box_color="#111c38", border_color=ACCENT_AMBER)

    draw_arrow(ax, 0.23, 0.42, 0.33, 0.54, color=ACCENT_CYAN)
    draw_arrow(ax, 0.23, 0.40, 0.33, 0.34, color=ACCENT_CYAN)
    draw_arrow(ax, 0.23, 0.38, 0.33, 0.14, color=ACCENT_CYAN)

    # Stage 3: Cross-Attention Fusion Transformer
    draw_box(ax, 0.71, 0.24, 0.26, 0.28, "Cross-Attention Fusion", "Q = Viseme, K = Audio, V = Temporal\nAttention = Softmax(QK^T / √d) V\nAV Sync Delay Latency (Δτ_av)\nBoundary Seam Score S_spatial", box_color="#0b1736", border_color=ACCENT_BLUE)

    draw_arrow(ax, 0.61, 0.54, 0.71, 0.42, color=ACCENT_SKY)
    draw_arrow(ax, 0.61, 0.34, 0.71, 0.38, color=ACCENT_PURPLE)
    draw_arrow(ax, 0.61, 0.14, 0.71, 0.30, color=ACCENT_AMBER)

    # Stage 4: Classification & Audit
    draw_box(ax, 1.07, 0.26, 0.26, 0.24, "Video Verdict & XAI", "P(Fake) = σ(0.35·S_sp + 0.30·S_spec +\n0.25·S_temp + 0.10·S_audio - 0.50)\nVerdict: REAL / MANIPULATED\nAutomatic Local Disk Preservation", box_color="#062e26", border_color=ACCENT_GREEN, title_color="#34d399")

    draw_arrow(ax, 0.97, 0.38, 1.07, 0.38, color=ACCENT_GREEN, text="Fused")

    ax.text(0.7, 0.02, "© 2026 Dar Rehan Rasool (@darrehanrasool) • Supervisors: Dr. Riyaz* & Dr. Javad* (IUST) • DeepFake-AI", ha="center", va="center", color=TEXT_MUTED, fontsize=8, fontfamily="monospace")

    out_path = OUTPUT_DIR / "video_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"[✓] Generated: {out_path}")


# ==============================================================================
# 3. IMAGE ARCHITECTURE DIAGRAM
# ==============================================================================
def generate_image_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 0.75)
    ax.axis("off")

    # Header
    ax.text(0.7, 0.71, "Multi-Scale Spatial-Frequency Image Forensics Architecture", ha="center", va="center", color=TEXT_MAIN, fontsize=14, fontweight="heavy")
    ax.text(0.7, 0.67, "Laplacian Convolutional Seam Analysis & 2D-DCT Power Spectrum Decomposition", ha="center", va="center", color=ACCENT_CYAN, fontsize=9.5, fontweight="bold")

    # Stage 1: Input Image
    draw_box(ax, 0.05, 0.28, 0.18, 0.24, "Input Face / Image", "Live Selfie / Photo Upload\nRGB (256 × 256 Normalized)\nGAN / Diffusion / DSLR\nCenter & Perimeter Sampling", box_color="#0b1329", border_color=ACCENT_CYAN, title_color=ACCENT_CYAN)

    # Stage 2: Dual Decomposition (Spatial vs Frequency)
    # Stream A: Spatial Boundary Seam Detection
    draw_box(ax, 0.33, 0.44, 0.28, 0.18, "Spatial Boundary Stream", "Discrete Laplacian Kernel ∇²I:\n[ 0  1  0;  1 -4  1;  0  1  0 ]\nEdge Gradient Variance (σ_edge²)\nColor Boundary Delta Δ_color", box_color="#111c38", border_color=ACCENT_SKY)
    # Stream B: 2D-DCT Power Spectrum
    draw_box(ax, 0.33, 0.18, 0.28, 0.18, "Frequency Spectral Stream", "2D Discrete Cosine Transform\nF(u,v) = α(u)α(v) Σ I(x,y) cos(...) cos(...)\nAzimuthal Integration: P(f)\n1/f² Power Law Anomaly Check", box_color="#111c38", border_color=ACCENT_PURPLE)

    draw_arrow(ax, 0.23, 0.44, 0.33, 0.53, color=ACCENT_CYAN)
    draw_arrow(ax, 0.23, 0.36, 0.33, 0.27, color=ACCENT_CYAN)

    # Stage 3: Dual-Stream Meso-Inception Fusion
    draw_box(ax, 0.71, 0.24, 0.26, 0.28, "Dual-Stream Fusion Network", "Spatial Score S_spatial ∈ [0, 1]\nSpectral Score S_spectral ∈ [0, 1]\nDiffusion Residual Checkerboard Net\nMeso-Inception Block Concatenation", box_color="#0b1736", border_color=ACCENT_BLUE)

    draw_arrow(ax, 0.61, 0.53, 0.71, 0.42, color=ACCENT_SKY)
    draw_arrow(ax, 0.61, 0.27, 0.71, 0.34, color=ACCENT_PURPLE)

    # Stage 4: Calibration & Explainability
    draw_box(ax, 1.07, 0.26, 0.26, 0.24, "Image Verdict & XAI", "P(Fake) = σ(0.55·S_sp + 0.45·S_spec - 0.40)\nConfidence Interval Calculation\nInteractive Heatmap / Anomaly Map\nPreservation in data/captured_tests/", box_color="#062e26", border_color=ACCENT_GREEN, title_color="#34d399")

    draw_arrow(ax, 0.97, 0.38, 1.07, 0.38, color=ACCENT_GREEN, text="z = logit")

    ax.text(0.7, 0.02, "© 2026 Dar Rehan Rasool (@darrehanrasool) • Supervisors: Dr. Riyaz* & Dr. Javad* (IUST) • DeepFake-AI", ha="center", va="center", color=TEXT_MUTED, fontsize=8, fontfamily="monospace")

    out_path = OUTPUT_DIR / "image_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"[✓] Generated: {out_path}")


# ==============================================================================
# 4. MASTER COMBINED SYSTEM ARCHITECTURE DIAGRAM
# ==============================================================================
def generate_master_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 0.9)
    ax.axis("off")

    # Header
    ax.text(0.8, 0.86, "DeepFake-AI™ Master Multi-Modal Cyber-Forensics Enterprise Architecture", ha="center", va="center", color=TEXT_MAIN, fontsize=15, fontweight="heavy")
    ax.text(0.8, 0.82, "Unified Real-Time Spatial, Spectral, Temporal & Acoustic Authenticity Verification Framework", ha="center", va="center", color=ACCENT_CYAN, fontsize=10, fontweight="bold")

    # Column 1: Multi-Modal Ingestion Layer (x = 0.05)
    ax.text(0.16, 0.76, "1. INGESTION STUDIO", ha="center", va="center", color=ACCENT_SKY, fontsize=9.5, fontweight="bold")
    draw_box(ax, 0.05, 0.58, 0.22, 0.14, "Live Video Recording", "Browser MediaRecorder API\nVideo + Audio Simultaneous Stream\nWebM / MP4 H.264 / VP9", box_color="#0b1329", border_color=ACCENT_CYAN)
    draw_box(ax, 0.05, 0.40, 0.22, 0.14, "Live Webcam Selfie", "getUserMedia Camera Viewfinder\nPhoto Capture & Center Alignment\nResolution: 640x480, JPEG 95%", box_color="#0b1329", border_color=ACCENT_CYAN)
    draw_box(ax, 0.05, 0.22, 0.22, 0.14, "Live Microphone Audio", "Web Audio API ScriptProcessor\nPCM 16 kHz, 16-bit Mono WAV\nDynamic RMS Volume Meter", box_color="#0b1329", border_color=ACCENT_CYAN)
    draw_box(ax, 0.05, 0.04, 0.22, 0.14, "File Upload / Presets", "Drag & Drop Multi-Format Demux\nBenchmark Presets: DFDC, GAN,\nTacotron Voice Clones", box_color="#0b1329", border_color=ACCENT_CYAN)

    # Column 2: Preprocessing & Demuxing (x = 0.35)
    ax.text(0.46, 0.76, "2. PREPROCESSING & DEMUX", ha="center", va="center", color=ACCENT_SKY, fontsize=9.5, fontweight="bold")
    draw_box(ax, 0.35, 0.50, 0.22, 0.20, "Video & Visual Pipeline", "MTCNN Face Alignment (224x224)\nTemporal Sequence Splitting (8 fps)\nFace Landmark Landmark Coordinates\nSpatial RGB Normalization", box_color="#0d1b3a", border_color=ACCENT_BLUE)
    draw_box(ax, 0.35, 0.16, 0.22, 0.20, "Acoustic Signal Pipeline", "16 kHz Nyquist Resampling\nHanning Window Framing (25 ms)\nSTFT Spectrogram Matrix\nZero-Crossing & Pitch Tremor", box_color="#0d1b3a", border_color=ACCENT_BLUE)

    # Arrows Col 1 -> Col 2
    draw_arrow(ax, 0.27, 0.65, 0.35, 0.62, color=ACCENT_CYAN)
    draw_arrow(ax, 0.27, 0.47, 0.35, 0.58, color=ACCENT_CYAN)
    draw_arrow(ax, 0.27, 0.29, 0.35, 0.26, color=ACCENT_CYAN)
    draw_arrow(ax, 0.27, 0.11, 0.35, 0.22, color=ACCENT_CYAN)

    # Column 3: Quad-Domain Feature Extraction (x = 0.65)
    ax.text(0.77, 0.76, "3. QUAD-DOMAIN FORENSIC EXTRACTORS", ha="center", va="center", color=ACCENT_SKY, fontsize=9.5, fontweight="bold")
    draw_box(ax, 0.65, 0.58, 0.24, 0.14, "S_spatial: Boundary Seam", "Laplacian Kernel ∇²I Discontinuity\nCenter-to-Perimeter Color Delta\nArtifact Seam Blending Score", box_color="#111c38", border_color=ACCENT_SKY)
    draw_box(ax, 0.65, 0.40, 0.24, 0.14, "S_spectral: 2D-DCT Power", "2D Discrete Cosine Transform\nAzimuthal Power Spectrum Roll-off\nGAN/Diffusion Grid Checkerboard", box_color="#111c38", border_color=ACCENT_PURPLE)
    draw_box(ax, 0.65, 0.22, 0.24, 0.14, "S_temporal: 3D-CNN & EAR", "3D-ResNet Motion Trajectory\nEye Aspect Ratio Derivative dE/dt\nPhoneme-to-Viseme Lip Delay", box_color="#111c38", border_color=ACCENT_AMBER)
    draw_box(ax, 0.65, 0.04, 0.24, 0.14, "S_audio: Vocoder Flatness", "Active-Bin Spectral Flatness (SFM)\nHigh-Frequency Ratio (fc = 3kHz)\nSpeech Phase Dispersion Anomaly", box_color="#111c38", border_color=ACCENT_RED)

    # Arrows Col 2 -> Col 3
    draw_arrow(ax, 0.57, 0.62, 0.65, 0.65, color=ACCENT_SKY)
    draw_arrow(ax, 0.57, 0.58, 0.65, 0.47, color=ACCENT_PURPLE)
    draw_arrow(ax, 0.57, 0.54, 0.65, 0.29, color=ACCENT_AMBER)
    draw_arrow(ax, 0.57, 0.26, 0.65, 0.11, color=ACCENT_RED)

    # Column 4: Cross-Attention Fusion (x = 0.97)
    ax.text(1.09, 0.76, "4. ST2-SYNCNET FUSION", ha="center", va="center", color=ACCENT_SKY, fontsize=9.5, fontweight="bold")
    draw_box(ax, 0.97, 0.22, 0.24, 0.46, "Cross-Attention Transformer", "Trained on Dual Tesla T4 GPUs\nCorpus: 9,100 Sequences (7,280/1,820)\n\nAttention(Q, K, V) = Softmax(QK^T / √d) V\n\nEnsemble Logit Formulation:\nz = Σ w_i · S_i + b\nP(Fake) = 1 / (1 + e^(-z))\n\nValidation AUC: 0.9995\nAccuracy: 99.45% | EER: 0.52%", box_color="#081b3b", border_color=ACCENT_BLUE)

    # Arrows Col 3 -> Col 4
    draw_arrow(ax, 0.89, 0.65, 0.97, 0.55, color=ACCENT_SKY)
    draw_arrow(ax, 0.89, 0.47, 0.97, 0.48, color=ACCENT_PURPLE)
    draw_arrow(ax, 0.89, 0.29, 0.97, 0.40, color=ACCENT_AMBER)
    draw_arrow(ax, 0.89, 0.11, 0.97, 0.32, color=ACCENT_RED)

    # Column 5: Presentation & Storage (x = 1.29)
    ax.text(1.41, 0.76, "5. XAI & PRESERVATION", ha="center", va="center", color=ACCENT_SKY, fontsize=9.5, fontweight="bold")
    draw_box(ax, 1.29, 0.48, 0.25, 0.22, "Decision & XAI Beacon", "Neon Glow Strobe (Red/Green)\nWeb Audio Alarm & Harmonic Chime\nStep-by-Step Mathematical Trace\nDaubert Legal Admissibility Report", box_color="#062e26", border_color=ACCENT_GREEN, title_color="#34d399")
    draw_box(ax, 1.29, 0.16, 0.25, 0.24, "Storage & Audit Engine", "Local File Preservation:\ndata/captured_tests/ (img/aud/vid)\nmanifest.json Audit History\nDataset Log Terminal Streaming\nInstant Re-Test & Download", box_color="#181e2b", border_color=ACCENT_CYAN, title_color=ACCENT_CYAN)

    # Arrows Col 4 -> Col 5
    draw_arrow(ax, 1.21, 0.50, 1.29, 0.58, color=ACCENT_GREEN)
    draw_arrow(ax, 1.21, 0.35, 1.29, 0.28, color=ACCENT_CYAN)

    # Footer Attribution
    ax.text(0.8, 0.02, "© 2026 Dar Rehan Rasool (@darrehanrasool) • Supervisors: Dr. Riyaz* & Dr. Javad* (IUST) • DeepFake-AI Enterprise Platform", ha="center", va="center", color=TEXT_MUTED, fontsize=8.5, fontfamily="monospace")

    out_path = OUTPUT_DIR / "deepfake_ai_master_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"[✓] Generated: {out_path}")


if __name__ == "__main__":
    print("[*] Generating 4 Research-Grade Publication Architecture Diagrams...")
    generate_audio_diagram()
    generate_video_diagram()
    generate_image_diagram()
    generate_master_diagram()
    print("[★] All 4 publication diagrams successfully created in assets/images/!")
