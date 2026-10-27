"""XceptionNet backbone for deepfake boundary detection (Rossler et al., 2019)."""

import torch
import torch.nn as nn


class DeepfakeXception(nn.Module):
    """Lightweight Xception proxy with depthwise separable convolutions."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Depthwise Separable Block
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
