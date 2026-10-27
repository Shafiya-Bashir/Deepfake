"""Model explainability, Grad-CAM heatmaps, and forensic attribution reports."""

from deepfake_ai.explainability.gradcam import GradCAM
from deepfake_ai.explainability.feature_importance import extract_feature_importance
from deepfake_ai.explainability.explanation_reports import generate_forensic_report

__all__ = [
    'GradCAM',
    'extract_feature_importance',
    'generate_forensic_report',
]
