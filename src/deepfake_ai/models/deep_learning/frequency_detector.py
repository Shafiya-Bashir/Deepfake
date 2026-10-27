"""Dual-stream Spatial and Frequency Cross-Attention Network."""

import torch
import torch.nn as nn


class DualStreamFrequencyNet(nn.Module):
    """Fuses RGB spatial stream with 2D-DCT frequency stream."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Spatial stream
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Spectral stream
        self.freq_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_spatial: torch.Tensor, x_freq: torch.Tensor = None) -> torch.Tensor:
        if x_freq is None:
            x_freq = x_spatial
        fs = self.spatial_stream(x_spatial)
        ff = self.freq_stream(x_freq)
        fused = torch.cat([torch.flatten(fs, 1), torch.flatten(ff, 1)], dim=1)
        return self.classifier(fused)
