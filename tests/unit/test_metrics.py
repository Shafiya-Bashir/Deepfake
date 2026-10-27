import numpy as np
import pytest
from deepfake_ai.evaluation.metrics import compute_deepfake_metrics, compute_eer


@pytest.mark.unit
def test_metrics_computation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_deepfake_metrics(y_true, y_prob)
    assert metrics['roc_auc'] == 1.0
    assert metrics['accuracy'] == 1.0
    assert metrics['eer'] == 0.0
