"""EfficientNet deepfake detection network."""

import torch
import torch.nn as nn

if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    class SiLU(nn.Module):
        def __init__(self, inplace: bool = False):
            super().__init__()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)


class DeepfakeEfficientNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
