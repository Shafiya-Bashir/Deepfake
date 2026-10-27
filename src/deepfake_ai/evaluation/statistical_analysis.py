"""Statistical significance testing (Wilcoxon and DeLong tests)."""

from scipy import stats
import numpy as np


def compute_p_value(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    stat, p_val = stats.wilcoxon(scores_a, scores_b)
    return float(p_val)
