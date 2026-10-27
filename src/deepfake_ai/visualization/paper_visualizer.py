"""Visualizer calibrated for IEEE double-column papers."""

import matplotlib.pyplot as plt


def set_paper_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 300,
        'lines.linewidth': 2.0
    })
