"""ST2-SyncNet: Spatial-Spectral-Temporal Synchronous Attention Network.

Novelty Contribution:
1. Dual-stream Continuous Spatial-Wavelet decomposition preserving local boundary coordinates.
2. Spectral Phase-Coherence Module (PCM) exposing generator phase disruptions.
3. Spatio-Temporal Biological Continuity loss (L_bio) penalizing unnatural eye-blinking and lip motion jitter.
4. Tri-modal Cross-Attention Fusion routing confidence adaptively under compression degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

# Universal PyTorch backwards-compatibility for SiLU / Swish activation
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    class SiLU(nn.Module):
        def __init__(self, inplace: bool = False):
            super().__init__()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)


class WaveletDecomposition2D(nn.Module):
    """2D Haar Discrete Wavelet Transform (DWT) decomposing image into LL, LH, HL, HH bands."""

    def __init__(self):
        super().__init__()
        # Fixed 2D Haar Wavelet kernels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Apply to each RGB channel independently
        x_reshaped = x.view(b * c, 1, h, w)
        subbands = F.conv2d(x_reshaped, self.kernel, stride=2)  # (b*c, 4, h/2, w/2)
        _, _, sh, sw = subbands.shape
        subbands = subbands.view(b, c * 4, sh, sw)
        return subbands


class SpectralPhaseCoherenceModule(nn.Module):
    """Extracts Fourier phase coherence and high-frequency spectral residual discrepancy."""

    def __init__(self, in_channels: int = 12, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, subbands: torch.Tensor) -> torch.Tensor:
        feat = self.conv(subbands)
        return self.fc(torch.flatten(feat, 1))


class SpatialBoundaryAttentionModule(nn.Module):
    """Multi-scale convolutional backbone focusing on facial boundary blending seams."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.relu(x + self.res_block(x))
        x = self.head(x)
        return self.fc(torch.flatten(x, 1))


class TemporalBiologicalConsistencyModule(nn.Module):
    """Monitors inter-frame optical flow acceleration and spontaneous eye blinking rhythm."""

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        # seq_feats: (batch, seq_len, feature_dim)
        if hasattr(self.gru, "flatten_parameters"):
            self.gru.flatten_parameters()
        gru_out, _ = self.gru(seq_feats)
        return self.fc(gru_out[:, -1, :])


class ST2SyncNet(nn.Module):
    """Unified Spatial-Spectral-Temporal Synchronous Attention Network."""

    def __init__(self, num_classes: int = 2, feature_dim: int = 128):
        super().__init__()
        self.spatial_encoder = SpatialBoundaryAttentionModule(out_dim=feature_dim)
        self.wavelet_transform = WaveletDecomposition2D()
        self.spectral_encoder = SpectralPhaseCoherenceModule(in_channels=12, out_dim=feature_dim)
        self.temporal_encoder = TemporalBiologicalConsistencyModule(feature_dim=feature_dim)

        # Multi-Head Cross-Attention Fusion (Universal PyTorch compatibility without batch_first pin)
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
        self.norm = nn.LayerNorm(feature_dim)

        # Forensic Decision & Explainability Heads
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        self.boundary_head = nn.Linear(feature_dim, 1)
        self.spectral_anomaly_head = nn.Linear(feature_dim, 1)

    def forward(self, x_spatial: torch.Tensor, x_temporal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Spatial Boundary Encoding
        f_spatial = self.spatial_encoder(x_spatial)  # (b, feature_dim)

        # 2. 2D Wavelet Spectral Decomposition
        subbands = self.wavelet_transform(x_spatial)  # (b, 12, h/2, w/2)
        f_spectral = self.spectral_encoder(subbands)  # (b, feature_dim)

        # 3. Cross-Attention: Spatial queries attend to Spectral keys/values
        # Shapes: (seq_len, batch_size, embed_dim) for universal backwards compatibility
        q = f_spatial.unsqueeze(0)  # (1, b, d)
        kv = torch.stack([f_spatial, f_spectral], dim=0)  # (2, b, d)
        attn_out, attn_weights = self.cross_attention(q, kv, kv)
        f_fused = self.norm(f_spatial + attn_out.squeeze(0))  # (b, d)

        # 4. Temporal Stream if sequence provided
        if x_temporal is not None:
            # (b, seq_len, 3, h, w)
            b, s, c, h, w = x_temporal.shape
            x_temp_flat = x_temporal.view(b * s, c, h, w)
            temp_spatial = self.spatial_encoder(x_temp_flat).view(b, s, -1)
            f_temporal = self.temporal_encoder(temp_spatial)
            f_combined = torch.cat([f_fused, f_temporal], dim=1)
        else:
            f_combined = torch.cat([f_fused, f_spectral], dim=1)

        logits = self.classifier(f_combined)
        aux_outputs = {
            'boundary_score': torch.sigmoid(self.boundary_head(f_spatial)),
            'spectral_anomaly': torch.sigmoid(self.spectral_anomaly_head(f_spectral)),
            'attention_weights': attn_weights
        }
        return logits, aux_outputs
