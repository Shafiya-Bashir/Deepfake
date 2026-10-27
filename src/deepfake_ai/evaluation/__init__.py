"""Evaluation suite, metrics calculation, and benchmark comparisons."""

from deepfake_ai.evaluation.metrics import compute_deepfake_metrics, compute_eer
from deepfake_ai.evaluation.evaluation import Evaluator
from deepfake_ai.evaluation.model_comparison import compare_models
from deepfake_ai.evaluation.literature_benchmarks import compare_with_literature

__all__ = [
    'compute_deepfake_metrics',
    'compute_eer',
    'Evaluator',
    'compare_models',
    'compare_with_literature',
]
