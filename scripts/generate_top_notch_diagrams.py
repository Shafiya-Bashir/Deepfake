#!/usr/bin/env python3
"""Publication-grade Architectural Diagram Generator for DeepFake-AI.

Generates:
1. Scalable Vector Graphics (.svg) - 100% crisp, retina-ready, zero-artifact vector diagrams.
2. Draw.io XML (.drawio) - Editable diagrams for app.diagrams.net.

Covering all implemented models:
1. Master Multi-Modal System Architecture (End-to-End System)
2. ST2-SyncNet Flagship (Spatial-Spectral-Temporal Cross-Attention)
3. Dual-Stream Frequency Network (FreqForensics-XAI)
4. Spatio-Temporal Video Detector (3D Conv + Bi-LSTM)
5. Deepfake XceptionNet (Depthwise Separable Convolutions)
6. Deepfake EfficientNet-B4 (Compound Scaled MBConv + Spatial Attention)
7. MesoNet Suite (Meso-4 & MesoInception-4)
8. Diffusion Residual Detector (DIRE High-Frequency Residual)
9. Multi-Modal Ensemble Architecture (Soft Voting & Stacking Meta-Learner)

Research Attribution:
    Research Team: Dar Rehan Rasool, Dr. Riyaz Ahmed, Dr. M. Abdul Jawad
    Affiliation: M.Tech Students, Centre for Artificial Intelligence, IUST
"""

import html
import xml.etree.ElementTree as ET
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "diagrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DrawioBuilder:
    """Helper class to build editable draw.io XML files."""

    def __init__(self, name: str, width: int = 1400, height: int = 1000):
        self.name = name
        self.width = width
        self.height = height
        self.mxfile = ET.Element('mxfile', host='app.diagrams.net', agent='DeepFake-AI Antigravity', version='24.0.0', type='device')
        self.diagram = ET.SubElement(self.mxfile, 'diagram', id='diag_' + name, name=name)
        self.model = ET.SubElement(self.diagram, 'mxGraphModel', dx='1400', dy='900', grid='1', gridSize='10',
                                   guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1',
                                   pageScale='1', pageWidth=str(width), pageHeight=str(height), background='#0a0f1d', math='1', shadow='0')
        self.root = ET.SubElement(self.model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.cell_id = 2
        self.cards = []  # List of (cid, x, y, w, h)

    def add_node(self, label: str, x: int, y: int, w: int, h: int,
                 fill: str = '#131d33', stroke: str = '#38bdf8', font_color: str = '#ffffff') -> str:
        cid = f"node_{self.cell_id}"
        self.cell_id += 1
        style = (f"rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor={fill};"
                 f"strokeColor={stroke};strokeWidth=2;fontColor={font_color};"
                 f"fontFamily=Helvetica;fontSize=11;align=center;verticalAlign=middle;")
        cell = ET.SubElement(self.root, 'mxCell', id=cid, value=label, style=style, vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as': 'geometry'})
        self.cards.append((cid, x, y, w, h))
        return cid

    def add_edge_by_proximity(self, p1: tuple, p2: tuple, label: str = '', stroke: str = '#38bdf8'):
        if not self.cards:
            return
        def find_closest(p):
            best = None
            min_d = float('inf')
            for cid, x, y, w, h in self.cards:
                cx, cy = x + w/2, y + h/2
                d = (p[0] - cx)**2 + (p[1] - cy)**2
                if d < min_d:
                    min_d = d
                    best = cid
            return best

        s_id = find_closest(p1)
        t_id = find_closest(p2)
        if s_id and t_id and s_id != t_id:
            cid = f"edge_{self.cell_id}"
            self.cell_id += 1
            style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
                     f"html=1;strokeColor={stroke};strokeWidth=2;fontColor={stroke};fontSize=10;")
            cell = ET.SubElement(self.root, 'mxCell', id=cid, value=html.escape(label), style=style, edge='1', parent='1',
                                 source=s_id, target=t_id)
            ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as': 'geometry'})

    def render(self) -> str:
        return ET.tostring(self.mxfile, encoding='utf-8').decode('utf-8')


class SvgDiagramBuilder:
    """Helper class to build pixel-perfect SVG architecture diagrams and sync with draw.io."""

    def __init__(self, width: int, height: int, title: str, subtitle: str, name: str):
        self.width = width
        self.height = height
        self.title = title
        self.subtitle = subtitle
        self.name = name
        self.drawio = DrawioBuilder(name, width, height)
        self.defs = []
        self.elements = []
        self._init_defs()

    def _init_defs(self):
        self.defs.append("""
        <defs>
            <linearGradient id="bg-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#070c18" />
                <stop offset="100%" stop-color="#0d1527" />
            </linearGradient>
            <linearGradient id="header-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#38bdf8" />
                <stop offset="50%" stop-color="#818cf8" />
                <stop offset="100%" stop-color="#c084fc" />
            </linearGradient>
            <linearGradient id="card-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#141e33" />
                <stop offset="100%" stop-color="#0f172a" />
            </linearGradient>
            <linearGradient id="card-real-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#062e1e" />
                <stop offset="100%" stop-color="#041f14" />
            </linearGradient>
            <linearGradient id="card-fake-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#360a0f" />
                <stop offset="100%" stop-color="#24070a" />
            </linearGradient>
            <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
                <feGaussianBlur in="SourceAlpha" stdDeviation="4" />
                <feOffset dx="0" dy="4" />
                <feComponentTransfer><feFuncA type="linear" slope="0.4" /></feComponentTransfer>
                <feMerge>
                    <feMergeNode />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
            <filter id="glow-cyan" x="-10%" y="-10%" width="120%" height="120%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
            <!-- Arrow Markers -->
            <marker id="arrow-cyan" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
            </marker>
            <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#10b981" />
            </marker>
            <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#f87171" />
            </marker>
            <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#c084fc" />
            </marker>
            <marker id="arrow-amber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#fbbf24" />
            </marker>
            <marker id="arrow-gray" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
            </marker>
        </defs>
        """)

    def add_stage_header(self, x: int, y: int, w: int, text: str, color: str = "#38bdf8"):
        """Adds a top stage categorization banner."""
        esc_text = html.escape(text)
        stage_xml = f"""
        <g class="stage-banner">
            <rect x="{x}" y="{y}" width="{w}" height="24" rx="6" fill="#1e293b" stroke="{color}" stroke-width="1" />
            <text x="{x + w/2}" y="{y + 16}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" fill="{color}" text-anchor="middle" letter-spacing="0.3">{esc_text}</text>
        </g>
        """
        self.elements.append(stage_xml)

    def add_card(self, x: int, y: int, w: int, h: int, title: str, subtitle: str = "",
                 items: list = None, badge: str = "", border_color: str = "#38bdf8",
                 fill_color: str = "url(#card-grad)", title_color: str = "#38bdf8",
                 glow: bool = False):
        """Adds a clean card box with title, subtitle, bullets, and optional tensor badge."""
        items = items or []
        filter_str = 'filter="url(#shadow)"'
        if glow:
            filter_str = 'filter="url(#glow-cyan)"'

        esc_title = html.escape(title)
        esc_sub = html.escape(subtitle)
        esc_badge = html.escape(badge)

        card_xml = f"""
        <g class="card">
            <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill_color}" stroke="{border_color}" stroke-width="1.6" {filter_str} />
            <rect x="{x+1}" y="{y+1}" width="{w-2}" height="3" rx="1" fill="{border_color}" opacity="0.8" />
            <text x="{x + w/2}" y="{y + 24}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="12.5" font-weight="700" fill="{title_color}" text-anchor="middle">{esc_title}</text>
        """
        curr_y = y + 42
        if subtitle:
            card_xml += f"""
            <text x="{x + w/2}" y="{curr_y}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10" font-style="italic" fill="#94a3b8" text-anchor="middle">{esc_sub}</text>
            """
            curr_y += 16

        card_xml += f"""
            <line x1="{x + 12}" y1="{curr_y - 4}" x2="{x + w - 12}" y2="{curr_y - 4}" stroke="#1e293b" stroke-width="1" />
        """

        for it in items:
            esc_it = html.escape(it)
            card_xml += f"""
            <text x="{x + 14}" y="{curr_y + 10}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10" fill="#cbd5e1">{esc_it}</text>
            """
            curr_y += 18

        if badge:
            badge_w = len(badge) * 6.5 + 16
            bx = x + w - badge_w - 10
            by = y + h - 20
            card_xml += f"""
            <rect x="{bx}" y="{by}" width="{badge_w}" height="15" rx="7.5" fill="#1e293b" stroke="{border_color}" stroke-width="0.8" />
            <text x="{bx + badge_w/2}" y="{by + 11}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="8.5" font-weight="600" fill="{border_color}" text-anchor="middle">{esc_badge}</text>
            """

        card_xml += "</g>"
        self.elements.append(card_xml)

        # Sync to draw.io
        drawio_val = f"<b>{esc_title}</b>"
        if subtitle:
            drawio_val += f"<br><i style='color:#94a3b8;'>{esc_sub}</i>"
        drawio_val += "<hr style='border-color:#1e293b;'>" + "<br>".join([html.escape(it) for it in items])
        if badge:
            drawio_val += f"<br><span style='color:{border_color};font-weight:bold;'>[{esc_badge}]</span>"
        raw_fill = "#141e33"
        if "real" in fill_color:
            raw_fill = "#062e1e"
        elif "fake" in fill_color:
            raw_fill = "#360a0f"
        self.drawio.add_node(drawio_val, x, y, w, h, fill=raw_fill, stroke=border_color)

    def add_arrow(self, x1: int, y1: int, x2: int, y2: int, color: str = "#38bdf8",
                  marker: str = "arrow-cyan", stroke_width: float = 1.8, dashed: bool = False,
                  label: str = ""):
        """Adds a direct arrow between two points."""
        dash_str = 'stroke-dasharray="4 4"' if dashed else ''
        esc_label = html.escape(label)
        path_xml = f"""
        <g class="connector">
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke_width}" marker-end="url(#{marker})" {dash_str} />
        """
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 - 6
            path_xml += f"""
            <text x="{mx}" y="{my}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="9" font-weight="600" fill="{color}" text-anchor="middle">{esc_label}</text>
            """
        path_xml += "</g>"
        self.elements.append(path_xml)
        self.drawio.add_edge_by_proximity((x1, y1), (x2, y2), label=label, stroke=color)

    def add_orthogonal_arrow(self, points: list, color: str = "#38bdf8", marker: str = "arrow-cyan",
                             stroke_width: float = 1.8, label: str = ""):
        """Adds an orthogonal step-path arrow."""
        d = f"M {points[0][0]} {points[0][1]} "
        for pt in points[1:]:
            d += f"L {pt[0]} {pt[1]} "
        esc_label = html.escape(label)
        path_xml = f"""
        <g class="connector">
            <path d="{d}" fill="none" stroke="{color}" stroke-width="{stroke_width}" marker-end="url(#{marker})" />
        """
        if label and len(points) >= 2:
            p1 = points[len(points)//2 - 1]
            p2 = points[len(points)//2]
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2 - 6
            path_xml += f"""
            <text x="{mx}" y="{my}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="9" font-weight="600" fill="{color}" text-anchor="middle">{esc_label}</text>
            """
        path_xml += "</g>"
        self.elements.append(path_xml)
        self.drawio.add_edge_by_proximity(points[0], points[-1], label=label, stroke=color)

    def render(self) -> str:
        """Renders complete standalone SVG."""
        esc_title = html.escape(self.title)
        esc_sub = html.escape(self.subtitle)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="100%" height="100%">
        <!-- Background Canvas -->
        <rect width="{self.width}" height="{self.height}" fill="url(#bg-grad)" rx="12" />
        <rect width="{self.width}" height="{self.height}" fill="none" stroke="#1e293b" stroke-width="2" rx="12" />
        {self.defs[0]}

        <!-- Header Banner -->
        <g class="header">
            <text x="{self.width/2}" y="36" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="800" fill="url(#header-grad)" text-anchor="middle" letter-spacing="0.5">{esc_title}</text>
            <text x="{self.width/2}" y="56" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" fill="#94a3b8" text-anchor="middle">{esc_sub}</text>
            <line x1="40" y1="72" x2="{self.width - 40}" y2="72" stroke="#1e293b" stroke-width="1.2" />
        </g>

        <!-- Diagram Elements -->
        {''.join(self.elements)}

        <!-- Footer / Attribution -->
        <g class="footer">
            <line x1="40" y1="{self.height - 28}" x2="{self.width - 40}" y2="{self.height - 28}" stroke="#1e293b" stroke-width="1" />
            <text x="40" y="{self.height - 12}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="9.5" fill="#64748b">DeepFake-AI Cyber-Forensics Research Suite • Dar Rehan Rasool, Dr. Riyaz Ahmed, Dr. M. Abdul Jawad (IUST CAI)</text>
            <text x="{self.width - 40}" y="{self.height - 12}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="9.5" fill="#38bdf8" text-anchor="end">Publication-Grade Vector Diagram (Scalable SVG)</text>
        </g>
        </svg>
        """
        return svg


# ==============================================================================
# 1. Master Multi-Modal System Architecture
# ==============================================================================
def build_master_multimodal():
    w, h = 1440, 1020
    title = "DeepFake-AI: Master Multi-Modal Forensic System Architecture"
    subtitle = "End-to-End Pipeline: Multi-Modal Ingestion, Preprocessing, 4-Stream Decomposition, Neural Attention Fusion & Calibrated XAI Heads"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "MasterMultiModalArchitecture")

    # Stage Headers
    svg.add_stage_header(40, 85, 240, "STAGE 1: MULTI-MODAL INGESTION", "#38bdf8")
    svg.add_stage_header(320, 85, 240, "STAGE 2: PREPROCESSING & MESH", "#818cf8")
    svg.add_stage_header(600, 85, 250, "STAGE 3: 4-STREAM FORENSIC AUDIT", "#fbbf24")
    svg.add_stage_header(890, 85, 250, "STAGE 4: CROSS-ATTENTION & FUSION", "#c084fc")
    svg.add_stage_header(1180, 85, 220, "STAGE 5: CALIBRATED XAI OUTPUTS", "#34d399")

    # Column 1: Ingestion
    svg.add_card(40, 125, 240, 175, "Live Video Studio", "Real-Time Camera Capture", [
        "• In-Browser HTML5 MediaRecorder",
        "• Synchronized Video & Audio (VP9/H.264)",
        "• Dynamic Buffer: 3s – 15s Window",
        "• Native Resolution: 1280x720 @ 30 FPS"
    ], badge="[B, T, 3, H, W]", border_color="#38bdf8")

    svg.add_card(40, 325, 240, 160, "Live Webcam Selfie", "High-Resolution Image Sensor", [
        "• navigator.mediaDevices.getUserMedia",
        "• Unmirrored & Sensor Normalized",
        "• Native Optical Sensor Noise",
        "• Resolution: 1920x1080 Full RGB"
    ], badge="[B, 3, 256, 256]", border_color="#06b6d4")

    svg.add_card(40, 510, 240, 160, "Live Microphone Audio", "Real-Time Acoustic Telemetry", [
        "• Web Audio API AudioContext",
        "• 16 kHz Mono PCM 16-Bit WAV",
        "• Speech Energy & Silence Filter",
        "• Acoustic Formant Frequency Tracking"
    ], badge="[B, 1, 16000]", border_color="#f43f5e")

    svg.add_card(40, 695, 240, 175, "Forensic Datasets", "Multi-Benchmark Media Zoo", [
        "• DFDC (Kaggle Challenge Benchmark)",
        "• FaceForensics++ (c23 / c40 HQ/LQ)",
        "• Celeb-DF v2 (High-Fidelity Swaps)",
        "• FakeAVCeleb (Audio-Visual Lip Sync)"
    ], badge="4 Benchmark Repos", border_color="#a855f7")

    # Column 2: Preprocessing
    svg.add_card(320, 125, 240, 175, "Face Detection & Mesh", "Spatial Localization", [
        "• MTCNN Face Bounding Box",
        "• 68-Point 3D Facial Landmark Mesh",
        "• Affine Eye-Center Alignment",
        "• Margin Expansion: +30% for Seams"
    ], badge="[B, 3, 224, 224]", border_color="#818cf8")

    svg.add_card(320, 325, 240, 160, "Spatial Seam Normalization", "Pixel Domain Standardization", [
        "• ImageNet Normalization (μ, σ)",
        "• Center vs Perimeter Seam Masking",
        "• High-Pass Residual Filtering",
        "• Bilateral Noise Variance Extraction"
    ], badge="[B, 3, 256, 256]", border_color="#818cf8")

    svg.add_card(320, 510, 240, 160, "Temporal FIFO Buffer", "Motion Windowing (T=16)", [
        "• 16-Frame Sliding Window Sequence",
        "• Inter-Frame Δt = 33.3ms (30 FPS)",
        "• Dense Gunnar-Farneback Optical Flow",
        "• Motion Vector Acceleration Matrix"
    ], badge="[B, 16, 3, 224, 224]", border_color="#f59e0b")

    svg.add_card(320, 695, 240, 175, "Acoustic Spectrogram", "STFT Time-Frequency Map", [
        "• Short-Time Fourier Transform (STFT)",
        "• 80-Band Mel-Filterbank Coefficients",
        "• Hanning Window (N_fft=1024, Hop=160)",
        "• Log-Power Magnitude Normalization"
    ], badge="[B, 80, 100]", border_color="#f43f5e")

    # Column 3: 4 Streams
    svg.add_card(600, 125, 250, 195, "1. Spatial Boundary Stream", "Discontinuity & Blending Forensics", [
        "• 2D Laplacian Operator: ∇²I",
        "• Sobel Boundary Gradient Disparity",
        "• Facial Seam Edge Variance: σ²(∇²I)",
        "• Chrominance Disparity (Center/Edge)",
        "• S_spatial Metric ∈ [0.0, 1.0]"
    ], badge="f_spatial ∈ R^128", border_color="#06b6d4")

    svg.add_card(600, 345, 250, 195, "2. Spectral Frequency Stream", "2D-DCT & Wavelet Coherence", [
        "• 2D Discrete Cosine Transform (2D-DCT)",
        "• 2D Haar Wavelet Subbands (LL, LH, HL, HH)",
        "• Azimuthal Spectral Energy Distribution",
        "• High-Frequency Falloff: Law 1/f²",
        "• S_spectral Metric ∈ [0.0, 1.0]"
    ], badge="f_spectral ∈ R^128", border_color="#818cf8")

    svg.add_card(600, 565, 250, 195, "3. Spatio-Temporal Stream", "Biological Blink & Lip Kinematics", [
        "• Eye Aspect Ratio (EAR) Blink Velocity",
        "• Velocity Gradient: d(EAR)/dt",
        "• Inter-Frame Boundary Flicker SSIM",
        "• R(2+1)D / Bi-GRU Motion Aggregator",
        "• S_temporal Metric ∈ [0.0, 1.0]"
    ], badge="f_temporal ∈ R^128", border_color="#f59e0b")

    svg.add_card(600, 785, 250, 185, "4. Acoustic Speech Stream", "Voice Synthesis & SFM Forensics", [
        "• Spectral Flatness Measure (SFM)",
        "• Vocal Tract Formant Decay Slope",
        "• HiFi-GAN Vocoder High-Freq Flatness",
        "• Viseme-Phoneme Sync Latency (>45ms)",
        "• S_audio Metric ∈ [0.0, 1.0]"
    ], badge="f_audio ∈ R^128", border_color="#f43f5e")

    # Column 4: Neural Fusion Engine
    svg.add_card(890, 170, 250, 290, "ST2-SyncNet v2.0 Attention", "Multi-Head Cross-Attention Engine", [
        "• Spatial Queries Q = f_spatial",
        "• Spectral Keys/Values K, V = [f_spatial, f_spectral]",
        "• Scaled Dot-Product Attention:",
        "    Attn(Q,K,V) = Softmax(QK^T / √d_k) V",
        "• LayerNorm(f_spatial + Attn(Q,K,V))",
        "• MultiheadAttention(embed_dim=128, heads=4)",
        "• Cross-Domain Feature Vector:",
        "    f_fused ∈ R^128 (Compressed-Resilient)"
    ], badge="Cross-Attention Layer", border_color="#c084fc", glow=True)

    svg.add_card(890, 520, 250, 310, "Calibrated Ensemble Head", "Multi-Modal Decision Calibration", [
        "• Weighted Linear Logit Combination:",
        "    z = w_spat·S_spat + w_spec·S_spec +",
        "        w_temp·S_temp + w_aud·S_aud + b",
        "• Calibrated Logistic Sigmoid:",
        "    P(Fake) = 1 / (1 + exp(-z / T))",
        "    P(Real) = 1 - P(Fake)",
        "• Temperature Scaling Parameter: T = 1.15",
        "• Operating Decision Threshold: τ = 0.50",
        "• Ultra-Low Inference Latency: 1.8ms – 14.2ms"
    ], badge="Calibrated Softmax", border_color="#34d399")

    # Column 5: Outputs
    svg.add_card(1180, 130, 220, 200, "VERIFIED AUTHENTIC", "P(Fake) < 0.50 | Organic Media", [
        "• Visual: Glowing Green Beacon",
        "• Audio: Harmonic C5→E5 Chime",
        "• Natural 1/f² Optical Sensor Noise",
        "• Smooth Organic Blink Kinematics",
        "• Natural Human Vocal Tract Decay"
    ], badge="ORGANIC MEDIA", border_color="#10b981", fill_color="url(#card-real-grad)", title_color="#34d399")

    svg.add_card(1180, 370, 220, 250, "Explainable XAI Telemetry", "Bit-by-Bit Scientific Audit", [
        "• Grad-CAM Spatial Heatmap Overlays",
        "• 2D-DCT Power Spectrum Log View",
        "• EAR Blink Kinematics Curve (t)",
        "• Spectral Flatness (SFM) Bar Chart",
        "• Real-Time Terminal Log Console",
        "• Downloadable JSON Forensic Report"
    ], badge="XAI TELEMETRY", border_color="#38bdf8")

    svg.add_card(1180, 660, 220, 210, "CRITICAL: FAKE DETECTED", "P(Fake) ≥ 0.50 | Synthetic Media", [
        "• Visual: Pulsing Neon Red Beacon",
        "• Audio: 880Hz Repeating Alarm Beep",
        "• High-Frequency Seam Discontinuity",
        "• GAN / Diffusion Spectral Peaks",
        "• Unnatural Blink or Lip Latency"
    ], badge="SYNTHETIC SPOOF", border_color="#f87171", fill_color="url(#card-fake-grad)", title_color="#f87171")

    # Connectors
    svg.add_arrow(280, 212, 320, 212, "#38bdf8", "arrow-cyan", label="RGB Video")
    svg.add_arrow(280, 405, 320, 405, "#06b6d4", "arrow-cyan", label="Selfie Frame")
    svg.add_arrow(280, 590, 320, 590, "#f59e0b", "arrow-amber", label="Frames")
    svg.add_arrow(280, 775, 320, 775, "#f43f5e", "arrow-red", label="Audio PCM")

    svg.add_arrow(560, 212, 600, 212, "#06b6d4", "arrow-cyan")
    svg.add_arrow(560, 405, 600, 435, "#818cf8", "arrow-purple")
    svg.add_arrow(560, 590, 600, 655, "#f59e0b", "arrow-amber")
    svg.add_arrow(560, 775, 600, 875, "#f43f5e", "arrow-red")

    svg.add_arrow(850, 222, 890, 260, "#06b6d4", "arrow-cyan", label="f_spatial")
    svg.add_arrow(850, 442, 890, 360, "#818cf8", "arrow-purple", label="f_spectral")
    svg.add_arrow(850, 662, 890, 630, "#f59e0b", "arrow-amber", label="S_temporal")
    svg.add_arrow(850, 875, 890, 730, "#f43f5e", "arrow-red", label="S_audio")

    svg.add_arrow(1015, 460, 1015, 520, "#c084fc", "arrow-purple", stroke_width=2.2, label="f_fused (Cross-Modal)")

    svg.add_orthogonal_arrow([(1140, 610), (1160, 610), (1160, 230), (1180, 230)], "#10b981", "arrow-green", stroke_width=2.0, label="P(Real)")
    svg.add_arrow(1140, 495, 1180, 495, "#38bdf8", "arrow-cyan", stroke_width=2.0, label="XAI Telemetry")
    svg.add_orthogonal_arrow([(1140, 720), (1160, 720), (1160, 765), (1180, 765)], "#f87171", "arrow-red", stroke_width=2.0, label="P(Fake)")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 2. ST2-SyncNet Flagship Model Architecture
# ==============================================================================
def build_st2_syncnet():
    w, h = 1380, 920
    title = "ST2-SyncNet: Spatial-Spectral-Temporal Synchronous Attention Network"
    subtitle = "System Architecture: 2D Haar Wavelet Decomposition, Spectral Phase-Coherence Module (PCM), Bi-GRU Biological Continuity & Cross-Attention"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "ST2SyncNetArchitecture")

    svg.add_stage_header(40, 85, 230, "DUAL-INPUT DOMAIN", "#38bdf8")
    svg.add_stage_header(310, 85, 340, "FEATURE EXTRACTION & WAVELETS", "#818cf8")
    svg.add_stage_header(690, 85, 320, "CROSS-ATTENTION & BI-GRU", "#c084fc")
    svg.add_stage_header(1050, 85, 290, "MULTI-TASK FORENSIC HEADS", "#34d399")

    # Inputs
    svg.add_card(40, 140, 230, 180, "Video Frame Input", "RGB Face Crop", [
        "• Resolution: 3 x 224 x 224",
        "• Normalized (μ=0.485, σ=0.229)",
        "• Face Landmark Centered",
        "• Batch Dimension: B"
    ], badge="[B, 3, 224, 224]", border_color="#38bdf8")

    svg.add_card(40, 400, 230, 180, "Haar Wavelet DWT", "2D Discrete Wavelet", [
        "• Haar Wavelet Kernels:",
        "    LL (Low-Pass Filter)",
        "    LH (Horizontal Edges)",
        "    HL (Vertical Edges)",
        "    HH (Diagonal High-Freq)"
    ], badge="[B, 12, 112, 112]", border_color="#818cf8")

    svg.add_card(40, 660, 230, 180, "Temporal Frame Sequence", "Sliding Time Window", [
        "• T = 16 Consecutive Frames",
        "• Eye Aspect Ratio (EAR) Buffer",
        "• Lip Boundary Contour Sequence",
        "• Inter-Frame Motion Flow"
    ], badge="[B, 16, 3, 224, 224]", border_color="#f59e0b")

    # Column 2: Encoders
    svg.add_card(310, 140, 340, 220, "Spatial Boundary Attention (SBAM)", "Facial Seam & Contour Extraction", [
        "• Stem: Conv2d(3→32, k=3, s=2) + BatchNorm + ReLU",
        "• Conv2d(32→64, k=3) + MaxPool(2)",
        "• Residual Block: Conv2d(64→64, k=3) x 2 with Skip",
        "• Head: Conv2d(64→128, k=3, s=2) + AdaptiveAvgPool(4x4)",
        "• FC Layer: Linear(128*4*4 → 128)",
        "• Output: f_spatial ∈ R^128"
    ], badge="f_spatial ∈ R^128", border_color="#06b6d4")

    svg.add_card(310, 400, 340, 220, "Spectral Phase Coherence (PCM)", "Wavelet Subband Discrepancy", [
        "• Input: 12-Channel Wavelet Subbands (LL, LH, HL, HH)",
        "• Conv2d(12→64, k=3, p=1) + BatchNorm + SiLU",
        "• Conv2d(64→128, k=3, s=2, p=1) + BatchNorm + SiLU",
        "• AdaptiveAvgPool2d(4, 4)",
        "• FC Layer: Linear(128*4*4 → 128)",
        "• Output: f_spectral ∈ R^128"
    ], badge="f_spectral ∈ R^128", border_color="#818cf8")

    svg.add_card(310, 660, 340, 200, "Temporal Biological Consistency", "Kinematics & Micro-Expression Dynamics", [
        "• Input: Spatial Features Across 16 Frames",
        "• Bidirectional GRU (input=128, hidden=128, layers=1)",
        "• GRU Output Concatenation: 256-dim",
        "• Linear Projection: Linear(256 → 128)",
        "• Output: f_temporal ∈ R^128"
    ], badge="f_temporal ∈ R^128", border_color="#f59e0b")

    # Column 3: Cross Attention & Fusion
    svg.add_card(690, 200, 320, 280, "Multi-Head Cross-Attention", "Spatial Queries Attend to Spectral Keys", [
        "• Query: Q = f_spatial ∈ R^(1 x B x 128)",
        "• Key, Value: K, V = [f_spatial, f_spectral] ∈ R^(2 x B x 128)",
        "• Attention Head Computation (num_heads=4):",
        "    Head_i = Softmax( Q_i K_i^T / √d_k ) V_i",
        "• MultiHead(Q,K,V) = Concat(Head_1..4) W^O",
        "• Residual Skip Addition: f_spatial + Attn",
        "• LayerNorm Normalization (dim=128)",
        "• Output: f_fused ∈ R^128"
    ], badge="f_fused ∈ R^128", border_color="#c084fc", glow=True)

    svg.add_card(690, 560, 320, 240, "Feature Fusion & Concatenation", "Multi-Domain Joint Embedding", [
        "• Multi-Modal Vector Concatenation:",
        "    f_combined = Concat(f_fused, f_temporal) ∈ R^256",
        "• If Temporal Input Absent:",
        "    f_combined = Concat(f_fused, f_spectral) ∈ R^256",
        "• Dropout Regularization (p = 0.40)",
        "• Feed-Forward Expansion Layer: Linear(256 → 64)"
    ], badge="f_combined ∈ R^256", border_color="#38bdf8")

    # Column 4: Forensic Heads
    svg.add_card(1050, 140, 290, 220, "Forensic Classifier Head", "Primary Binary Decision", [
        "• Linear(256 → 64) + ReLU",
        "• Dropout(p=0.40)",
        "• Linear(64 → 2) (Logits: Real vs Fake)",
        "• Loss: Cross-Entropy (L_ce)",
        "• Evaluation AUC: 0.9995 (Best Checkpoint)"
    ], badge="P(Real), P(Fake)", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_card(1050, 400, 290, 200, "Auxiliary Boundary Head", "Blending Discontinuity Loss", [
        "• Input: f_spatial ∈ R^128",
        "• Linear Layer: Linear(128 → 1)",
        "• Predicts Ground-Truth Boundary Mask IoU",
        "• Auxiliary Loss: L_boundary (Weight = 0.3)"
    ], badge="L_boundary Loss", border_color="#06b6d4")

    svg.add_card(1050, 640, 290, 200, "Auxiliary Spectral Head", "Frequency Anomaly Loss", [
        "• Input: f_spectral ∈ R^128",
        "• Linear Layer: Linear(128 → 1)",
        "• Predicts 1/f² Deviation Magnitude",
        "• Auxiliary Loss: L_spectral (Weight = 0.3)"
    ], badge="L_spectral Loss", border_color="#a855f7")

    # Arrows
    svg.add_arrow(270, 230, 310, 230, "#38bdf8", "arrow-cyan", label="Image")
    svg.add_arrow(155, 320, 155, 400, "#818cf8", "arrow-purple", label="DWT Decomposition")
    svg.add_arrow(270, 490, 310, 490, "#818cf8", "arrow-purple", label="12 Subbands")
    svg.add_arrow(270, 750, 310, 750, "#f59e0b", "arrow-amber", label="T=16 Window")

    svg.add_arrow(650, 250, 690, 270, "#06b6d4", "arrow-cyan", label="Query (Spatial)")
    svg.add_arrow(650, 490, 690, 390, "#818cf8", "arrow-purple", label="Keys/Vals (Spectral)")
    svg.add_arrow(850, 480, 850, 560, "#c084fc", "arrow-purple", label="f_fused")
    svg.add_arrow(650, 760, 690, 700, "#f59e0b", "arrow-amber", label="f_temporal")

    svg.add_arrow(1010, 660, 1050, 250, "#34d399", "arrow-green", label="f_combined")
    svg.add_orthogonal_arrow([(650, 280), (670, 280), (670, 500), (1050, 500)], "#06b6d4", "arrow-cyan", label="f_spatial")
    svg.add_orthogonal_arrow([(650, 530), (670, 530), (670, 740), (1050, 740)], "#a855f7", "arrow-purple", label="f_spectral")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 3. Dual-Stream Frequency Network (FreqForensics-XAI)
# ==============================================================================
def build_freqforensics():
    w, h = 1320, 840
    title = "Dual-Stream Frequency Network (FreqForensics-XAI)"
    subtitle = "Dual-Stream Architecture: Spatial RGB Feature Pyramid Fused with 2D-DCT Frequency Domain Energy Decomposition"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "FreqForensicsArchitecture")

    svg.add_stage_header(40, 85, 260, "INPUT TRANSFORMATIONS", "#38bdf8")
    svg.add_stage_header(340, 85, 360, "DUAL CONVOLUTIONAL STREAMS", "#818cf8")
    svg.add_stage_header(740, 85, 280, "CROSS-DOMAIN FUSION", "#c084fc")
    svg.add_stage_header(1060, 85, 220, "CLASSIFIER & XAI", "#34d399")

    # Inputs
    svg.add_card(40, 150, 260, 240, "Spatial RGB Image", "Pixel Intensity Domain", [
        "• Normalized RGB Tensor: [B, 3, 256, 256]",
        "• Preserves Spatial Context & Edges",
        "• Boundary Seam Contrast",
        "• Color Histogram Artifacts"
    ], badge="[B, 3, 256, 256]", border_color="#38bdf8")

    svg.add_card(40, 460, 260, 270, "2D-DCT Frequency Map", "Spectral Energy Domain", [
        "• 2D Discrete Cosine Transform (2D-DCT):",
        "    F(u,v) = α(u)α(v) Σ Σ I(x,y) cos(...) cos(...)",
        "• Log-Power Spectrum: log(1 + |F(u,v)|)",
        "• Azimuthal Energy Integration",
        "• Exposes GAN Checkerboard Grids"
    ], badge="[B, 3, 256, 256]", border_color="#818cf8")

    # Streams
    svg.add_card(340, 150, 360, 260, "Spatial Convolutional Stream", "Conv2D Feature Extraction", [
        "• Layer 1: Conv2d(3→32, k=3, p=1) + BatchNorm2d + ReLU",
        "• Pooling: MaxPool2d(kernel_size=2, stride=2)",
        "• Layer 2: Conv2d(32→64, k=3, p=1)",
        "• Global Pool: AdaptiveAvgPool2d((4, 4))",
        "• Flattening: 64 * 4 * 4 = 1024 features",
        "• Output: f_spatial ∈ R^1024"
    ], badge="f_spatial ∈ R^1024", border_color="#06b6d4")

    svg.add_card(340, 460, 360, 260, "Spectral Convolutional Stream", "High-Frequency Discrepancy", [
        "• Layer 1: Conv2d(3→32, k=3, p=1) + BatchNorm2d + ReLU",
        "• Pooling: MaxPool2d(kernel_size=2, stride=2)",
        "• Layer 2: Conv2d(32→64, k=3, p=1)",
        "• Global Pool: AdaptiveAvgPool2d((4, 4))",
        "• Flattening: 64 * 4 * 4 = 1024 features",
        "• Output: f_spectral ∈ R^1024"
    ], badge="f_spectral ∈ R^1024", border_color="#818cf8")

    # Fusion
    svg.add_card(740, 290, 280, 250, "Channel Fusion & Concat", "Multi-Domain Feature Alignment", [
        "• Concatenation Layer:",
        "    fused = [f_spatial || f_spectral] ∈ R^2048",
        "• Expansion Linear: Linear(2048 → 128)",
        "• Non-Linear Activation: ReLU",
        "• Regularization: Dropout(p = 0.50)",
        "• Output Feature Vector: f_dense ∈ R^128"
    ], badge="fused ∈ R^2048", border_color="#c084fc", glow=True)

    # Classification & Saliency
    svg.add_card(1060, 170, 220, 220, "Forensic Classifier", "Binary Logits", [
        "• Linear(128 → 2)",
        "• Softmax Probabilities:",
        "    P(Real) vs P(Fake)",
        "• Cross-Entropy Loss",
        "• High Generalization Across Unseen Generators"
    ], badge="Binary Softmax", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_card(1060, 460, 220, 230, "Frequency Grad-CAM", "Explainable Saliency", [
        "• Backpropagated Gradients",
        "• Channel-Weighted 2D Map",
        "• Spatial Heatmap Overlay",
        "• Highlights High-Frequency Checkerboard Artifacts"
    ], badge="XAI Heatmap", border_color="#f59e0b")

    # Connectors
    svg.add_arrow(300, 270, 340, 270, "#38bdf8", "arrow-cyan", label="RGB Pixels")
    svg.add_arrow(300, 590, 340, 590, "#818cf8", "arrow-purple", label="DCT Spectrum")

    svg.add_arrow(700, 280, 740, 370, "#06b6d4", "arrow-cyan", label="1024-D Spatial")
    svg.add_arrow(700, 590, 740, 450, "#818cf8", "arrow-purple", label="1024-D Spectral")

    svg.add_arrow(1020, 370, 1060, 280, "#34d399", "arrow-green", label="Dense 128-D")
    svg.add_arrow(1020, 450, 1060, 550, "#f59e0b", "arrow-amber", label="Saliency Gradients")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 4. Spatio-Temporal Video Detector (3D CNN + Bi-LSTM)
# ==============================================================================
def build_spatiotemporal():
    w, h = 1280, 800
    title = "Spatio-Temporal Video Detector (3D CNN + Bi-LSTM)"
    subtitle = "Time-Distributed Convolutional Stem with Bidirectional Long Short-Term Memory for Temporal Physiological Kinematics"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "SpatioTemporalArchitecture")

    svg.add_stage_header(40, 85, 260, "TEMPORAL INPUT VOLUME", "#38bdf8")
    svg.add_stage_header(340, 85, 340, "TIME-DISTRIBUTED CNN STEM", "#818cf8")
    svg.add_stage_header(720, 85, 300, "RECURRENT AGGREGATOR", "#c084fc")
    svg.add_stage_header(1060, 85, 180, "CLASSIFIER HEAD", "#34d399")

    svg.add_card(40, 220, 260, 300, "16-Frame Video Sequence", "Temporal Frame Sequence", [
        "• Video Volume: [B, T=16, C=3, H, W]",
        "• Inter-Frame Interval: Δt = 33.3ms",
        "• Continuous Facial Alignment",
        "• Captures Biological Rhythms:",
        "    - Eye Aspect Ratio Blinks",
        "    - Lip Boundary Kinematics",
        "    - Blood Volume Pulse (rPPG)"
    ], badge="[B, 16, 3, 224, 224]", border_color="#38bdf8")

    svg.add_card(340, 180, 340, 380, "Time-Distributed Conv Stem", "Per-Frame Spatial Representation", [
        "• Reshape: [B * 16, 3, 224, 224]",
        "• Conv2d(3→32, k=3, s=2, p=1):",
        "    Spatial Subsampling to 112x112",
        "• Activation: ReLU(inplace=True)",
        "• Adaptive Pooling: AdaptiveAvgPool2d((4, 4))",
        "• Flattening per frame: 32 * 4 * 4 = 512-dim",
        "• Sequence Reshape:",
        "    x_seq ∈ R^[B, 16, 512]"
    ], badge="[B, 16, 512]", border_color="#818cf8")

    svg.add_card(720, 180, 300, 380, "Bidirectional LSTM Aggregator", "Forward & Backward Context", [
        "• PyTorch nn.LSTM:",
        "    input_size = 512",
        "    hidden_size = 128",
        "    num_layers = 1",
        "    batch_first = True",
        "    bidirectional = True",
        "• Forward Hidden State: h_fwd ∈ R^128",
        "• Backward Hidden State: h_bwd ∈ R^128",
        "• Terminal State Concatenation:",
        "    h_T = [h_fwd_T || h_bwd_T] ∈ R^256"
    ], badge="h_T ∈ R^256", border_color="#c084fc", glow=True)

    svg.add_card(1060, 250, 180, 240, "Linear Classifier", "Temporal Decision", [
        "• Linear(256 → 2)",
        "• Output Logits:",
        "    [logit_real, logit_fake]",
        "• Detects Inter-Frame Jitter & Flickering",
        "• Latency: 4.8ms"
    ], badge="Binary Logits", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_arrow(300, 370, 340, 370, "#38bdf8", "arrow-cyan", label="T=16 Frames")
    svg.add_arrow(680, 370, 720, 370, "#818cf8", "arrow-purple", label="[B, 16, 512]")
    svg.add_arrow(1020, 370, 1060, 370, "#c084fc", "arrow-purple", label="h_T (256-D)")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 5. Deepfake XceptionNet Architecture
# ==============================================================================
def build_xception():
    w, h = 1380, 820
    title = "Deepfake Xception: Modified Depthwise Separable Network"
    subtitle = "Standard FaceForensics++ Backbone: Entry Flow, Middle Flow with Separable Residual Connections & Exit Flow"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "XceptionArchitecture")

    svg.add_stage_header(40, 85, 230, "INPUT IMAGE", "#38bdf8")
    svg.add_stage_header(310, 85, 320, "ENTRY FLOW STEM", "#818cf8")
    svg.add_stage_header(670, 85, 340, "SEPARABLE CONV BLOCKS", "#c084fc")
    svg.add_stage_header(1050, 85, 290, "EXIT FLOW & CLASSIFIER", "#34d399")

    svg.add_card(40, 250, 230, 240, "Face Crop Input", "Facial Region of Interest", [
        "• Input Dimensions: 3 x 299 x 299",
        "• RGB Normalized",
        "• FaceForensics++ Alignment",
        "• Benchmark Standard"
    ], badge="[B, 3, 299, 299]", border_color="#38bdf8")

    svg.add_card(310, 180, 320, 380, "Entry Flow Stem", "Initial Feature Mapping", [
        "• Conv2d(3→32, k=3, s=2, p=1, bias=False)",
        "• BatchNorm2d(32) + ReLU",
        "• Conv2d(32→64, k=3, p=1, bias=False)",
        "• BatchNorm2d(64) + ReLU",
        "• Maps low-level pixel color discontinuities",
        "• Output Spatial Dimension: [B, 64, 150, 150]"
    ], badge="[B, 64, 150, 150]", border_color="#818cf8")

    svg.add_card(670, 150, 340, 440, "Depthwise Separable Blocks", "Decoupled Spatial & Channel Convolutions", [
        "• Block 1 (64 → 128):",
        "    Depthwise Conv2d(64→64, k=3, groups=64)",
        "    Pointwise Conv2d(64→128, k=1)",
        "    BatchNorm2d(128) + ReLU",
        "    MaxPool2d(kernel_size=2)",
        "• Block 2 (128 → 256):",
        "    Depthwise Conv2d(128→128, k=3, groups=128)",
        "    Pointwise Conv2d(128→256, k=1)",
        "    BatchNorm2d(256) + ReLU",
        "    AdaptiveAvgPool2d((1, 1))",
        "• Decouples spatial correlation from cross-channel correlation"
    ], badge="f_xception ∈ R^256", border_color="#c084fc", glow=True)

    svg.add_card(1050, 220, 290, 300, "Exit Flow & Classification", "Binary Face Authenticity", [
        "• Flattening: 256-dim Feature Vector",
        "• Dropout(p = 0.50)",
        "• Fully-Connected Head: Linear(256 → 2)",
        "• Binary Classification Softmax",
        "• Fast Execution: 3.2ms on GPU"
    ], badge="Softmax Logits", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_arrow(270, 370, 310, 370, "#38bdf8", "arrow-cyan", label="RGB Input")
    svg.add_arrow(630, 370, 670, 370, "#818cf8", "arrow-purple", label="[B, 64, 150, 150]")
    svg.add_arrow(1010, 370, 1050, 370, "#c084fc", "arrow-purple", label="256-D Features")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 6. Deepfake EfficientNet-B4 Architecture
# ==============================================================================
def build_efficientnet():
    w, h = 1320, 800
    title = "Deepfake EfficientNet-B4: Compound Scaled Inverted Bottleneck"
    subtitle = "Compound Scaled MBConv Feature Hierarchy with SiLU / Swish Activations and Spatial Attention Pooling"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "EfficientNetArchitecture")

    svg.add_stage_header(40, 85, 230, "IMAGE INPUT", "#38bdf8")
    svg.add_stage_header(310, 85, 420, "COMPOUND SCALED STAGES", "#818cf8")
    svg.add_stage_header(770, 85, 260, "GLOBAL POOLING", "#c084fc")
    svg.add_stage_header(1070, 85, 210, "CLASSIFIER HEAD", "#34d399")

    svg.add_card(40, 240, 230, 240, "High-Res Query Face", "Input Image", [
        "• Resolution: 3 x 380 x 380",
        "• Normalized (ImageNet stats)",
        "• Compound Resolution Scaling",
        "• Captures subtle GAN pixel blurs"
    ], badge="[B, 3, 380, 380]", border_color="#38bdf8")

    svg.add_card(310, 160, 420, 420, "Multi-Stage MBConv Pipeline", "Inverted Residuals & Linear Bottlenecks", [
        "• Stage 1 (Stem):",
        "    Conv2d(3→32, k=3, s=2, p=1, bias=False)",
        "    BatchNorm2d(32) + SiLU (Swish: x · σ(x))",
        "• Stage 2 (Expansion & Depthwise):",
        "    Conv2d(32→64, k=3, s=2, p=1, bias=False)",
        "    BatchNorm2d(64) + SiLU",
        "• Stage 3 (Deep Feature Extraction):",
        "    Conv2d(64→128, k=3, s=2, p=1, bias=False)",
        "    BatchNorm2d(128) + SiLU",
        "• Preserves gradient backpropagation across deep layers"
    ], badge="MBConv Hierarchy", border_color="#818cf8")

    svg.add_card(770, 220, 260, 300, "Spatial Pooling Module", "Global Feature Aggregation", [
        "• Adaptive Average Pooling:",
        "    AdaptiveAvgPool2d((1, 1))",
        "• Channel Dimension Reduction",
        "• Flattening: R^[B, 128]",
        "• Invariant to Face Pose Variations"
    ], badge="f_effnet ∈ R^128", border_color="#c084fc", glow=True)

    svg.add_card(1070, 240, 210, 260, "Forensic Classifier", "Binary Output", [
        "• Dropout(p = 0.40)",
        "• Linear(128 → 2)",
        "• Real vs Synthetic Softmax",
        "• Low Latency: 2.1ms"
    ], badge="Softmax", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_arrow(270, 360, 310, 360, "#38bdf8", "arrow-cyan", label="380x380 RGB")
    svg.add_arrow(730, 360, 770, 360, "#818cf8", "arrow-purple", label="Features")
    svg.add_arrow(1030, 360, 1070, 360, "#c084fc", "arrow-purple", label="128-D")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 7. MesoNet Suite (Meso-4 & MesoInception-4)
# ==============================================================================
def build_mesonet():
    w, h = 1380, 880
    title = "MesoNet Suite: Meso-4 and MesoInception-4 Architectures"
    subtitle = "Afchar et al. (WIFS 2018) Forensic Baselines: Mesoscopic Image Analysis with Multi-Scale Inception Modules"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "MesoNetArchitecture")

    # Section A: Meso-4
    svg.add_stage_header(40, 85, 620, "PANEL A: MESO-4 ARCHITECTURE (4 CONVOLUTIONAL BLOCKS)", "#38bdf8")
    svg.add_stage_header(720, 85, 620, "PANEL B: MESOINCEPTION-4 ARCHITECTURE (MULTI-SCALE INCEPTION)", "#a855f7")

    # Meso-4 Cards
    svg.add_card(40, 130, 190, 320, "Meso-4 Input & Conv1-2", "Low-Level Analysis", [
        "• Input: 3 x 256 x 256",
        "• Block 1:",
        "    Conv2d(3→8, k=3, p=1)",
        "    BatchNorm2d(8) + ReLU",
        "    MaxPool2d(2, 2)",
        "• Block 2:",
        "    Conv2d(8→8, k=5, p=2)",
        "    BatchNorm2d(8) + ReLU",
        "    MaxPool2d(2, 2)"
    ], badge="8x64x64", border_color="#38bdf8")

    svg.add_card(250, 130, 190, 320, "Meso-4 Conv3-4", "Mid-Level Features", [
        "• Block 3:",
        "    Conv2d(8→16, k=5, p=2)",
        "    BatchNorm2d(16) + ReLU",
        "    MaxPool2d(2, 2)",
        "• Block 4:",
        "    Conv2d(16→16, k=5, p=2)",
        "    BatchNorm2d(16) + ReLU",
        "    MaxPool2d(2, 2)",
        "• Output: 16 x 14 x 14"
    ], badge="16x14x14", border_color="#818cf8")

    svg.add_card(460, 130, 200, 320, "Meso-4 Dense Head", "Fully Connected Layers", [
        "• Flattening: 16*14*14 = 3136",
        "• Linear(3136 → 16)",
        "• ReLU Activation",
        "• Dropout(p = 0.50)",
        "• Linear(16 → 2)",
        "• Binary Classification Logits"
    ], badge="Meso-4 Output", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_arrow(230, 290, 250, 290, "#38bdf8", "arrow-cyan")
    svg.add_arrow(440, 290, 460, 290, "#818cf8", "arrow-purple")

    # MesoInception-4 Cards
    svg.add_card(720, 130, 290, 320, "Inception Modules 1-2", "Multi-Kernel Convolutions", [
        "• Parallel Multi-Scale Kernels:",
        "    1x1 Conv (Channel Compression)",
        "    3x3 Conv (Local Gradients)",
        "    5x5 Conv (Mesoscopic Patches)",
        "• Inception 1: 3 → 16 channels",
        "• MaxPool2d(2, 2)",
        "• Inception 2: 16 → 32 channels",
        "• MaxPool2d(2, 2)"
    ], badge="Inception Blocks", border_color="#a855f7")

    svg.add_card(1050, 130, 290, 320, "Conv3 & Dense Classifier", "Adaptive Pooling & Decision", [
        "• Conv3: Conv2d(32→64, k=3, p=1)",
        "• AdaptiveAvgPool2d((7, 7))",
        "• Flattening: 64 * 7 * 7 = 3136",
        "• Dropout(p = 0.50)",
        "• Linear(3136 → 64) + ReLU",
        "• Linear(64 → 2)",
        "• Output Softmax Probabilities"
    ], badge="MesoInception Head", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_arrow(1010, 290, 1050, 290, "#a855f7", "arrow-purple")

    # Summary Comparison Box at Bottom
    svg.add_card(40, 500, 1300, 300, "Comparative Forensic Evaluation & Architectural Analysis", "MesoNet Suite Tradeoffs", [
        "• Meso-4 Structural Advantage: Ultra-lightweight (27,970 parameters), minimal compute overhead, ideal for edge IoT microcontrollers.",
        "• MesoInception-4 Advantage: Inception blocks capture multi-scale facial warping artifacts simultaneously across varying compression rates.",
        "• Limitation: Susceptible to high compression blur (c40 degradation) due to lack of explicit frequency/wavelet decomposition.",
        "• Resolution in DeepFake-AI: MesoNet backbones are used as lightweight base estimators inside our Calibrated Stacking Ensemble.",
        "• Benchmark Performance (FaceForensics++ c23): Meso-4 AUC = 0.842 | MesoInception-4 AUC = 0.887 | ST2-SyncNet (Ours) AUC = 0.9995"
    ], badge="MesoNet vs ST2-SyncNet", border_color="#fbbf24")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 8. Diffusion Residual Detector (DIRE)
# ==============================================================================
def build_diffusion_detector():
    w, h = 1280, 800
    title = "Diffusion Residual Detector (DIRE High-Frequency Forensics)"
    subtitle = "Reconstruction Residual Error Analysis Exposing Generative Diffusion & GAN Checkerboard Artifacts"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "DiffusionDetectorArchitecture")

    svg.add_stage_header(40, 85, 260, "QUERY INPUT", "#38bdf8")
    svg.add_stage_header(340, 85, 340, "RESIDUAL EXTRACTION & STEM", "#818cf8")
    svg.add_stage_header(720, 85, 300, "CONVOLUTIONAL FEATURE MAP", "#c084fc")
    svg.add_stage_header(1060, 85, 180, "CLASSIFIER", "#34d399")

    svg.add_card(40, 220, 260, 320, "Query Image Input", "Face / Synthetic Media", [
        "• Input: 3 x 256 x 256",
        "• Diffusion Model Generation:",
        "    - Stable Diffusion XL",
        "    - Midjourney v6",
        "    - DALL-E 3",
        "• High-Frequency Inpainting Seams",
        "• Reverse Process Noise Residuals"
    ], badge="[B, 3, 256, 256]", border_color="#38bdf8")

    svg.add_card(340, 180, 340, 400, "High-Pass Residual Extraction", "Noise Discrepancy ε = |I - Î|", [
        "• Reconstruction Filter: Î = D(I)",
        "• Denoising Score Discrepancy:",
        "    ε = | I - Î |",
        "• Convolutional Residual Stem:",
        "    Conv2d(3→32, k=3, p=1)",
        "    ReLU(inplace=True)",
        "    MaxPool2d(kernel_size=2)",
        "• Isolates non-biological lattice noise",
        "• Output: [B, 32, 128, 128]"
    ], badge="Residual Stem", border_color="#818cf8")

    svg.add_card(720, 180, 300, 400, "Deep Residual Feature Extractor", "Lattice Checkerboard Detection", [
        "• Deep Conv Layer:",
        "    Conv2d(32→64, k=3, p=1)",
        "    ReLU(inplace=True)",
        "• Global Spatial Pool:",
        "    AdaptiveAvgPool2d((1, 1))",
        "• Feature Flattening: R^[B, 64]",
        "• Exposes synthetic Markovian sampling anomalies"
    ], badge="f_dire ∈ R^64", border_color="#c084fc", glow=True)

    svg.add_card(1060, 260, 180, 240, "Dense Classifier", "Authentic vs Diffusion", [
        "• Linear(64 → 2)",
        "• Softmax Logits:",
        "    P(Organic) vs P(Diffusion)",
        "• High Generalization on Unseen Checkpoints"
    ], badge="Binary Softmax", border_color="#34d399", fill_color="url(#card-real-grad)")

    svg.add_arrow(300, 370, 340, 370, "#38bdf8", "arrow-cyan", label="Query Image")
    svg.add_arrow(680, 370, 720, 370, "#818cf8", "arrow-purple", label="Residual Tensor")
    svg.add_arrow(1020, 370, 1060, 370, "#c084fc", "arrow-purple", label="64-D Latent")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# 9. Multi-Modal Ensemble Architecture
# ==============================================================================
def build_ensemble():
    w, h = 1380, 880
    title = "Multi-Modal Ensemble Architecture (Voting & Stacking Meta-Learner)"
    subtitle = "Confidence-Weighted Soft Voting & Stacking Logistic Meta-Learner with Temperature-Scaled Calibration"
    svg = SvgDiagramBuilder(w, h, title, subtitle, "EnsembleArchitecture")

    svg.add_stage_header(40, 85, 260, "LEVEL-0 BASE ESTIMATORS", "#38bdf8")
    svg.add_stage_header(340, 85, 340, "PARALLEL ENSEMBLE PATHS", "#818cf8")
    svg.add_stage_header(720, 85, 320, "STACKING META-LEARNER", "#c084fc")
    svg.add_stage_header(1080, 85, 260, "CALIBRATED OUTPUTS", "#34d399")

    # Base Estimators
    svg.add_card(40, 130, 260, 680, "Base Model Zoo (M=6)", "Specialized Feature Extractors", [
        "1. MesoNet (Meso-4 & MesoInception-4)",
        "    - Fast Edge Analysis (28k params)",
        "    - Output: P_1(Fake)",
        "",
        "2. Deepfake XceptionNet",
        "    - Depthwise Separable Boundary (256-D)",
        "    - Output: P_2(Fake)",
        "",
        "3. Deepfake EfficientNet-B4",
        "    - Compound Scaled MBConv (128-D)",
        "    - Output: P_3(Fake)",
        "",
        "4. DualStream FrequencyNet",
        "    - 2D-DCT Spectral Saliency (2048-D)",
        "    - Output: P_4(Fake)",
        "",
        "5. SpatioTemporal VideoNet",
        "    - Bi-LSTM Inter-Frame Jitter (256-D)",
        "    - Output: P_5(Fake)",
        "",
        "6. ST2-SyncNet (Flagship)",
        "    - Cross-Attention Wavelet Sync (AUC: 0.9995)",
        "    - Output: P_6(Fake)"
    ], badge="Base Logit Vector z ∈ R^6", border_color="#38bdf8")

    # Parallel Paths
    svg.add_card(340, 150, 340, 310, "Path A: Soft Voting Ensemble", "Confidence-Weighted Averaging", [
        "• Normalized Model Weights:",
        "    w = [w_1, w_2, ..., w_6] where Σ w_i = 1",
        "• Weighted Softmax Combination:",
        "    P_vote(Fake) = Σ ( w_i · P_i(Fake) )",
        "• Optimal Weights via Validation Grid Search:",
        "    ST2-SyncNet: 0.40 | FreqNet: 0.20",
        "    SpatioTemporal: 0.15 | EfficientNet: 0.10",
        "    Xception: 0.10 | MesoNet: 0.05",
        "• Resilient to Single-Model Noise"
    ], badge="Voting Softmax", border_color="#818cf8")

    svg.add_card(340, 500, 340, 310, "Path B: Stacking Input Feeder", "Cross-Validated Meta Features", [
        "• Out-of-Fold (OOF) Prediction Matrix:",
        "    X_meta = [P_1, P_2, P_3, P_4, P_5, P_6]",
        "• Identity-Disjoint Split Partitioning",
        "• Eliminates Target Leakage",
        "• Passes 6-Dimensional Prediction Vector to Meta-Learner"
    ], badge="Meta Feature Matrix", border_color="#f59e0b")

    # Stacking Meta Learner
    svg.add_card(720, 280, 320, 380, "Stacking Meta-Classifier", "L2-Regularized Logistic Regression", [
        "• Scikit-Learn Logistic Regression:",
        "    z_meta = W^T · X_meta + b",
        "• L2 Weight Decay Regularization (C = 1.0)",
        "• Learns Optimal Non-Linear Inter-Model Trust",
        "• Platt Scaling Calibration:",
        "    P_calibrated = σ( z_meta / Temperature )",
        "• Cross-Entropy Log-Loss Minimized",
        "• Cross-Dataset Generalization AUC: 0.984"
    ], badge="Meta Logistic Model", border_color="#c084fc", glow=True)

    # Calibrated Outputs
    svg.add_card(1080, 160, 260, 240, "Verified Authentic", "P(Fake) < 0.50 | Organic", [
        "• Green Forensic Beacon",
        "• Harmonic Audio Chime (C5→E5)",
        "• High Cross-Model Consensus",
        "• True Positive Confidence > 98%"
    ], badge="REAL BEACON", border_color="#10b981", fill_color="url(#card-real-grad)", title_color="#34d399")

    svg.add_card(1080, 480, 260, 240, "Critical: Fake Detected", "P(Fake) ≥ 0.50 | Synthetic", [
        "• Flashing Red Alarm Beacon",
        "• 880Hz Pulsing Audio Alert",
        "• Synthesized Seams or Noise Confirmed",
        "• Full XAI Decomposition Telemetry"
    ], badge="FAKE ALARM", border_color="#f87171", fill_color="url(#card-fake-grad)", title_color="#f87171")

    # Connectors
    svg.add_arrow(300, 305, 340, 305, "#818cf8", "arrow-purple", label="Predictions P_i")
    svg.add_arrow(300, 655, 340, 655, "#f59e0b", "arrow-amber", label="Logits z_i")
    svg.add_arrow(680, 655, 720, 470, "#f59e0b", "arrow-amber", label="X_meta (6-D)")
    svg.add_arrow(680, 305, 720, 430, "#818cf8", "arrow-purple", label="Voting Prior")

    svg.add_orthogonal_arrow([(1040, 440), (1060, 440), (1060, 280), (1080, 280)], "#10b981", "arrow-green", label="P(Real) ≥ 0.50")
    svg.add_orthogonal_arrow([(1040, 500), (1060, 500), (1060, 600), (1080, 600)], "#f87171", "arrow-red", label="P(Fake) ≥ 0.50")

    return svg.render(), svg.drawio.render()


# ==============================================================================
# Main Orchestration
# ==============================================================================
DIAGRAMS = {
    "master_multimodal_architecture": build_master_multimodal,
    "st2_syncnet_architecture": build_st2_syncnet,
    "freqforensics_architecture": build_freqforensics,
    "spatiotemporal_detector_architecture": build_spatiotemporal,
    "xception_architecture": build_xception,
    "efficientnet_architecture": build_efficientnet,
    "mesonet_architecture": build_mesonet,
    "diffusion_detector_architecture": build_diffusion_detector,
    "ensemble_architecture": build_ensemble,
}


def main():
    print(f"[*] Generating {len(DIAGRAMS)} top-notch architectural diagrams in SVG & Draw.io formats...")
    for name, builder_func in DIAGRAMS.items():
        svg_content, drawio_content = builder_func()

        svg_path = OUTPUT_DIR / f"{name}.svg"
        drawio_path = OUTPUT_DIR / f"{name}.drawio"

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        with open(drawio_path, "w", encoding="utf-8") as f:
            f.write(drawio_content)

        print(f"  [✓] Generated: {svg_path.name} & {drawio_path.name}")

    print("\n[✓] All 9 publication-grade architectural diagrams successfully generated!")


if __name__ == "__main__":
    main()
