import numpy as np
import pytest
from deepfake_ai.pipeline.unified_pipeline import UnifiedPipeline


@pytest.mark.unit
def test_pipeline_prediction():
    pipeline = UnifiedPipeline()
    sample = np.zeros((224, 224, 3), dtype=np.uint8)
    res = pipeline.predict_image(sample)
    assert 'verdict' in res
    assert 'fake_probability' in res
