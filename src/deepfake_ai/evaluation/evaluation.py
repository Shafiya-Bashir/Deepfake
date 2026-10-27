"""Model evaluator over test dataloaders."""

import torch
import numpy as np
from typing import Dict, Any
from deepfake_ai.evaluation.metrics import compute_deepfake_metrics


class Evaluator:
    @staticmethod
    def evaluate_model(model: torch.nn.Module, dataloader, device: str = 'cpu') -> Dict[str, float]:
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                labels = batch['label'].cpu().numpy()
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_labels.extend(labels)
                all_probs.extend(probs)

        return compute_deepfake_metrics(np.array(all_labels), np.array(all_probs))
