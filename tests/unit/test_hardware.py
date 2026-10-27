import pytest
from deepfake_ai.hardware.hardware_detector import detect_hardware


@pytest.mark.unit
def test_hardware_detector():
    profile = detect_hardware()
    assert profile.os_name is not None
    assert profile.recommended_device in ['cuda', 'mps', 'cpu']
