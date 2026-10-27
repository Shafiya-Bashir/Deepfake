"""2D Discrete Cosine Transform (DCT) and spatial frequency feature extraction."""

import numpy as np
from scipy.fftpack import dct


class FrequencyFeatureExtractor:
    """Extracts 2D-DCT spectral energy maps and high-pass residuals."""

    @staticmethod
    def compute_2d_dct(image_gray: np.ndarray) -> np.ndarray:
        """Computes 2D-DCT type-II transform on grayscale image."""
        return dct(dct(image_gray.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def extract_highpass_residual(image_gray: np.ndarray, cutoff: int = 16) -> np.ndarray:
        """Isolates high-frequency generator artifacts via high-pass masking."""
        freq = FrequencyFeatureExtractor.compute_2d_dct(image_gray)
        h, w = freq.shape
        mask = np.ones((h, w), dtype=bool)
        mask[:cutoff, :cutoff] = False
        residual = freq * mask
        return residual
