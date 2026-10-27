import numpy as np
import pytest
from deepfake_ai.data.preprocessing import FacePreprocessor
from deepfake_ai.data.feature_engineering import FrequencyFeatureExtractor


@pytest.mark.unit
def test_face_crop():
    prep = FacePreprocessor()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = prep.crop_and_align(img, [100, 100, 200, 200])
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


@pytest.mark.unit
def test_2d_dct():
    img_gray = np.random.rand(64, 64)
    freq = FrequencyFeatureExtractor.compute_2d_dct(img_gray)
    assert freq.shape == (64, 64)
