"""Comparative bar charts for benchmark models."""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_model_comparison_bars(models: list, aucs: list, output_path: Path):
    plt.figure(figsize=(9, 5))
    plt.bar(models, aucs, color='#38bdf8')
    plt.ylim(0.5, 1.02)
    plt.title('Benchmark Models Comparative AUC-ROC')
    plt.ylabel('AUC')
    plt.xticks(rotation=20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
