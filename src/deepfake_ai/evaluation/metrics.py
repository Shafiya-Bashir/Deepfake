"""Metrics evaluation: AUC-ROC, Equal Error Rate (EER), F1, Accuracy, LogLoss."""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support, log_loss, brier_score_loss
from typing import Dict, Any


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Equal Error Rate (operating point where FPR == FNR)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def compute_deepfake_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes full suite of forensic performance metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    eer = compute_eer(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        'roc_auc': round(auc, 4),
        'eer': round(eer, 4),
        'accuracy': round(acc, 4),
        'precision': round(float(p), 4),
        'recall': round(float(r), 4),
        'f1_score': round(float(f1), 4),
        'log_loss': round(ll, 4),
        'brier_score': round(brier, 4),
    }
