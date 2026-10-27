"""Normalized confusion matrix heatmap plotting."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, output_path: Path, labels=['REAL', 'FAKE']):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
