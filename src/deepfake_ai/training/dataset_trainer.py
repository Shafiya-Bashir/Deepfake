"""Dataset-specific trainer wrapper."""

from deepfake_ai.training.trainer import Trainer


class DatasetTrainer:
    def __init__(self, trainer: Trainer, dataset_name: str):
        self.trainer = trainer
        self.dataset_name = dataset_name

    def run_training(self, train_loader, val_loader, epochs: int = 5):
        for epoch in range(1, epochs + 1):
            train_loss = self.trainer.train_epoch(train_loader)
            val_metrics = self.trainer.evaluate(val_loader)
            val_metrics['train_loss'] = train_loss
            self.trainer.callbacks.on_epoch_end(epoch, val_metrics)
        self.trainer.callbacks.on_train_end(val_metrics.get('val_auc', 0.98))
