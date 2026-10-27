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


@pytest.mark.unit
def test_st2_syncnet_forward():
    model = ModelFactory.create('st2_syncnet')
    x = torch.randn(2, 3, 224, 224)
    out, aux = model(x)
    assert out.shape == (2, 2)
    assert 'boundary_score' in aux
    assert 'spectral_anomaly' in aux


@pytest.mark.unit
def test_st2_syncnet_temporal_forward():
    from deepfake_ai.models.deep_learning.st2_syncnet import ST2SyncNet
    model = ST2SyncNet(num_classes=2, feature_dim=128)
    x_spatial = torch.randn(2, 3, 224, 224)
    x_temporal = torch.randn(2, 5, 3, 224, 224)  # 5 frames sequence
    out, aux = model(x_spatial, x_temporal)
    assert out.shape == (2, 2)
    assert 'boundary_score' in aux
    assert 'spectral_anomaly' in aux
    assert 'attention_weights' in aux
