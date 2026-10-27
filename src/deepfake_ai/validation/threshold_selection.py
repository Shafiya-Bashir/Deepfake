"""Optimal operating point selection via Youden's J index."""

import numpy as np
from sklearn.metrics import roc_curve


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])
