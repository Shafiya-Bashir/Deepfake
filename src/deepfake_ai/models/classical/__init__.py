"""Classical statistical machine learning models for deepfake detection."""

from deepfake_ai.models.classical.svm import build_svm
from deepfake_ai.models.classical.random_forest import build_random_forest
from deepfake_ai.models.classical.gradient_boosting import build_gradient_boosting
from deepfake_ai.models.classical.logistic_regression import build_logistic_regression

__all__ = [
    'build_svm',
    'build_random_forest',
    'build_gradient_boosting',
    'build_logistic_regression',
]
