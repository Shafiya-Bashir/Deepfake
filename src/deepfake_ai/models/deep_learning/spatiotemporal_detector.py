"""Spatio-Temporal Video Detector capturing inter-frame physiological cues."""

import torch
import torch.nn as nn


class SpatioTemporalDetector(nn.Module):
    """ResNet feature extractor combined with Bidirectional LSTM."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.lstm = nn.LSTM(
            input_size=32 * 4 * 4,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected x: (Batch, Seq_len, C, H, W) or (Batch, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        b, s, c, h, w = x.shape
        x_flat = x.view(b * s, c, h, w)
        feats = self.cnn(x_flat).view(b, s, -1)
        lstm_out, _ = self.lstm(feats)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
