#!/usr/bin/env python3
"""Generate high-resolution scientific system architecture diagram for DeepFake-AI.

Renders publication-quality dark-mode schematic flowchart with exact mathematical
formulations, multi-modal ingestion streams, neural attention fusion, and explainable heads.

Research Attribution:
    Faculty Supervisors: Dr. Riyaz & Dr. Javad (IUST)
    Research Contributor & Graduate Student: Dar Rehan Rasool
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATHS = [
    PROJECT_ROOT / "docs" / "paper" / "figures" / "system_architecture_diagram.png",
    PROJECT_ROOT / "assets" / "images" / "deepfake_ai_architecture.png"
]


def draw_architecture():
    fig = plt.figure(figsize=(18, 11), facecolor="#0a0f1d")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0a0f1d")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Research Attribution Header
    ax.text(50, 96.5, "DeepFake-AI: Unified Multi-Modal Spatial-Frequency & Temporal Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#38bdf8")
    ax.text(50, 93.8, "Research Team: Dar Rehan Rasool (Graduate Student & Research Contributor) | Dr. Riyaz & Dr. Javad (Faculty Supervisors, IUST)",
            ha="center", va="center", fontsize=11, color="#94a3b8")

    # Helper function to draw glowing cards
    def draw_box(x, y, w, h, title, subtitle="", items=None, border_color="#38bdf8", fill_color="#131d33", title_color="#38bdf8"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                             linewidth=1.8, edgecolor=border_color, facecolor=fill_color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 2.2, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=title_color)
        if subtitle:
            ax.text(x + w/2, y + h - 4.5, subtitle, ha="center", va="center", fontsize=8.2, color="#94a3b8", style="italic")
        if items:
            curr_y = y + h - 7.0
            for it in items:
                ax.text(x + 2.0, curr_y, it, ha="left", va="center", fontsize=8.0, color="#e2e8f0")
                curr_y -= 2.6

    # -------------------------------------------------------------
    # Tier 1: Multi-Modal Ingestion Streams (Top Layer)
    # -------------------------------------------------------------
    draw_box(4, 76, 26, 14, "1. Live & Uploaded Media Ingestion", "Multi-Modal Input Feeds", [
        "• Video Frames: .mp4, .mov, .avi (DFDC / FF++)",
        "• Live Webcam Selfie: In-Browser HTML5 Canvas",
        "• Audio / Voice: .wav, .mp3, .flac (FakeAVCeleb)",
        "• Live Microphone: In-Browser 16kHz PCM WAV"
    ], border_color="#38bdf8", title_color="#38bdf8")

    # MTCNN & Preprocessing Dispatcher
    draw_box(36, 76, 26, 14, "2. Face Detection & Feature Extraction", "Preprocessing & Normalization", [
        "• MTCNN Face Bounding Box & Landmark Mesh",
        "• Facial Contour Alignment (256x256 RGB)",
        "• Multi-Frame Window Buffer (T = 16 frames)",
        "• STFT / Mel-Spectrogram Acoustic Transform"
    ], border_color="#818cf8", title_color="#818cf8")

    # Connecting Arrow: Ingestion -> Preprocessing
    ax.annotate("", xy=(36, 83), xytext=(30, 83),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.2))

    # Saved Checkpoint Engine
    draw_box(68, 76, 28, 14, "3. Checkpoint & Pretrained Model Zoo", "State Dict Loading & PyTorch Weights", [
        "• Checkpoints: st2_syncnet_best.pt (AUC: 0.996)",
        "• Backbones: EfficientNet-B4 / Xception",
        "• Multi-GPU DataParallel (NVIDIA Tesla T4)",
        "• Strict Actor Identity-Disjoint Splits"
    ], border_color="#fbbf24", title_color="#fbbf24")

    # -------------------------------------------------------------
    # Tier 2: 4 Forensic Analysis Streams (Middle Layer)
    # -------------------------------------------------------------
    # Stream A: Spatial Domain
    draw_box(3, 47, 21.5, 22, "Spatial Boundary Stream", "S_spatial: Gradient Discontinuity", [
        "• 2D Laplacian Operator: ∇²I",
        "• Sobel Boundary Filtering",
        "• Edge Variance: σ²(∇²I)",
        "• Facial Seam Color Disparity",
        "• Blending Mask Edge Noise",
        "• High-Pass Residual Analysis"
    ], border_color="#06b6d4", title_color="#06b6d4")

    # Stream B: Spectral Domain
    draw_box(27.5, 47, 21.5, 22, "Spectral Frequency Stream", "S_spectral: 2D-DCT & Wavelets", [
        "• 2D Discrete Cosine Transform",
        "• Haar Wavelet Decomposition (LL, LH, HL, HH)",
        "• 1/f² Natural Image Fall-off Law",
        "• GAN/Diffusion Checkerboard Peaks",
        "• Azimuthal Spectral Energy Ratio",
        "• Transposed Conv Artifacts"
    ], border_color="#818cf8", title_color="#818cf8")

    # Stream C: Temporal Biological Stream
    draw_box(52, 47, 21.5, 22, "Temporal Biological Stream", "S_temporal: Spatio-Temporal Net", [
        "• Eye Aspect Ratio (EAR) Blinks",
        "• Dynamic Velocity d(EAR)/dt",
        "• Inter-Frame SSIM Delta Flow",
        "• Optical Flow Velocity Field",
        "• Biological rPPG Pulse Rhythm",
        "• Temporal Boundary Jitter"
    ], border_color="#f59e0b", title_color="#f59e0b")

    # Stream D: Acoustic Voice Stream
    draw_box(76.5, 47, 20.5, 22, "Acoustic Speech Stream", "S_audio: Voice Clone Forensics", [
        "• Spectral Flatness Measure (SFM)",
        "• Vocal Tract Formant Decay",
        "• Pitch Micro-Tremor (Jitter)",
        "• HiFi-GAN Vocoder Noise Floor",
        "• Audio-Visual Lip-Sync Jitter",
        "• Viseme-Phoneme Latency (>45ms)"
    ], border_color="#f43f5e", title_color="#f43f5e")

    # Connecting arrows from Preprocessing down to the 4 streams
    for target_x in [13.75, 38.25, 62.75, 86.75]:
        ax.annotate("", xy=(target_x, 69), xytext=(49, 76),
                    arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.6, connectionstyle="angle,angleA=0,angleB=90,rad=3"))

    # -------------------------------------------------------------
    # Tier 3: Neural Fusion & Decision Ensemble (Lower Layer)
    # -------------------------------------------------------------
    draw_box(20, 23, 60, 16, "Multi-Head Cross-Attention Fusion (ST2-SyncNet) & Calibrated Ensemble",
             "Multi-Modal Logistic Calibrated Regression Engine", [
        "  z = w_spatial · S_spatial + w_spectral · S_spectral + w_temporal · S_temporal + w_audio · S_audio + b",
        "  P(Fake) = 1 / (1 + e^(-z))    |    P(Real) = 1 - P(Fake)    |    Threshold: τ = 0.50",
        "  Attention Mechanism: MultiheadAttention(embed_dim=128, heads=4) attends Spatial Queries to Spectral Keys",
        "  Online Inference Latency: 1.8ms - 14.2ms | Hardware: Apple Silicon MPS / Dual Tesla T4 GPUs"
    ], border_color="#22c55e", title_color="#22c55e")

    # Arrows from 4 streams down to Fusion Engine
    for src_x in [13.75, 38.25, 62.75, 86.75]:
        ax.annotate("", xy=(50, 39), xytext=(src_x, 47),
                    arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.6))

    # -------------------------------------------------------------
    # Tier 4: Explainable Decision Outputs & Interactive Alerts
    # -------------------------------------------------------------
    # Real Alert Box
    draw_box(4, 3, 27, 14, "VERIFIED AUTHENTIC", "P(Fake) < 0.50 | Organic Media", [
        "• Visual Light: Glowing Neon GREEN Beacon",
        "• Auditory Alert: Harmonic C5->E5 Chime",
        "• Natural 1/f² Optical Sensor Decay",
        "• Natural Human Vocal Tract Formants"
    ], border_color="#22c55e", fill_color="#062817", title_color="#4ade80")

    # Explainability Saliency Box
    draw_box(35, 3, 30, 14, "Explainability (XAI) Dashboard", "Traceability & Bit-by-Bit Telemetry", [
        "• Grad-CAM & Frequency Heatmap Overlays",
        "• 4-Step Mathematical Trace Breakdown",
        "• Live VS Code Log Console (logs/ directory)",
        "• In-Browser Webcam & Microphone Studio"
    ], border_color="#38bdf8", fill_color="#0a192f", title_color="#38bdf8")

    # Fake Alert Box
    draw_box(69, 3, 27, 14, "CRITICAL: FAKE DETECTED", "P(Fake) >= 0.50 | Synthetic Media", [
        "• Visual Light: Flashing Neon RED Beacon",
        "• Auditory Alert: Pulsing 880Hz Alarm Beep",
        "• GAN/Diffusion Checkerboard Detected",
        "• Neural Vocoder Noise Flatness Verified"
    ], border_color="#ef4444", fill_color="#2d080a", title_color="#f87171")

    # Arrows from Fusion down to 3 output panels
    ax.annotate("", xy=(17.5, 17), xytext=(35, 23),
                arrowprops=dict(arrowstyle="->", color="#22c55e", lw=2.2))
    ax.annotate("", xy=(50, 17), xytext=(50, 23),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.2))
    ax.annotate("", xy=(82.5, 17), xytext=(65, 23),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.2))

    plt.tight_layout()
    for out_path in OUTPUT_PATHS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"[✓] Architecture Diagram saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    draw_architecture()
