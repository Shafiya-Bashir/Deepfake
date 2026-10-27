"""Precision-Recall curve visualization."""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path


def plot_precision_recall_curve(y_true, y_score, output_path: Path):
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(7, 6))
    plt.plot(r, p, color='#ff7f0e', lw=2.5, label=f'PR Curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
