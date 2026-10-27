"""Epoch training loss and validation AUC curves."""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(train_losses, val_aucs, output_path: Path):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='#d62728')
    ax1.plot(epochs, train_losses, color='#d62728', lw=2, label='Train Loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val AUC', color='#1f77b4')
    ax2.plot(epochs, val_aucs, color='#1f77b4', lw=2, label='Val AUC')

    plt.title('Training Loss and Validation AUC Progression')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
