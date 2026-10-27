"""Facial bounding box extraction, margin expansion, and affine alignment."""

import numpy as np
import torch
from typing import Tuple, List, Optional


class FacePreprocessor:
    """Standardizes face detection and resizing."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224), margin: float = 1.3):
        self.target_size = target_size
        self.margin = margin

    def crop_and_align(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crops face with contextual margin."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = (x2 - x1) * self.margin, (y2 - y1) * self.margin

        nx1 = max(0, int(cx - bw / 2))
        ny1 = max(0, int(cy - bh / 2))
        nx2 = min(w, int(cx + bw / 2))
        ny2 = min(h, int(cy + bh / 2))

        crop = image[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=image.dtype)
        return crop
