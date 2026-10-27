"""Error analysis plots breakdown."""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_error_breakdown(fp_count: int, fn_count: int, output_path: Path):
    plt.figure(figsize=(5, 4))
    plt.bar(['False Positives', 'False Negatives'], [fp_count, fn_count], color=['#ff7f0e', '#d62728'])
    plt.ylabel('Count')
    plt.title('Error Type Breakdown')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
