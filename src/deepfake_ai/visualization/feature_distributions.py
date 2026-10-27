"""Feature distribution plots for real vs synthetic samples."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_feature_distributions(df, feature_name: str, output_path: Path):
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x=feature_name, hue='label_name', common_norm=False, palette={'REAL': '#2ca02c', 'FAKE': '#d62728'})
    plt.title(f'Feature Distribution: {feature_name}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
