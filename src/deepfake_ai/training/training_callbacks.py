"""Training callbacks for logging and alerts."""

import logging
from deepfake_ai.monitoring.audio_alert import play_alert

logger = logging.getLogger(__name__)


class CallbackHandler:
    def on_epoch_end(self, epoch: int, metrics: dict):
        logger.info(f'Epoch {epoch:02d} Summary: ' + ' | '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in metrics.items()))

    def on_train_end(self, best_metric: float):
        logger.info(f'Training pipeline concluded. Peak metric: {best_metric:.4f}')
        play_alert('complete')
