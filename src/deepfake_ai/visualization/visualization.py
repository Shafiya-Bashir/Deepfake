"""Main Visualizer dispatcher class."""

from pathlib import Path
from deepfake_ai.visualization.roc_curves import plot_roc_curve
from deepfake_ai.visualization.confusion_matrix import plot_confusion_matrix


class Visualizer:
    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_evaluation_plots(self, y_true, y_score, prefix: str = 'eval'):
        plot_roc_curve(y_true, y_score, self.output_dir / f'{prefix}_roc.png')
        y_pred = (y_score >= 0.5).astype(int)
        plot_confusion_matrix(y_true, y_pred, self.output_dir / f'{prefix}_confusion.png')
