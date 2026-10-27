import torch
import pytest
from deepfake_ai.models.model_factory import ModelFactory


@pytest.mark.unit
def test_mesonet_forward():
    model = ModelFactory.create('mesoinception4')
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


@pytest.mark.unit
def test_xception_forward():
    model = ModelFactory.create('xception')
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


@pytest.mark.unit
def test_dual_stream_forward():
    model = ModelFactory.create('dual_stream_frequency')
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
