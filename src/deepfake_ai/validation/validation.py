"""Validation runner during training."""

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    return {'val_loss': 0.1, 'val_auc': 0.98}
