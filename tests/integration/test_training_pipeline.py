import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deepfake_ai.training.trainer import Trainer
from deepfake_ai.models.deep_learning.mesonet import Meso4


@pytest.mark.integration
def test_mini_training_loop():
    model = Meso4()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    trainer = Trainer(model, opt, crit)

    # 4 synthetic samples
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    ds = [{'image': x[i], 'label': y[i]} for i in range(4)]
    loader = DataLoader(ds, batch_size=2)

    loss = trainer.train_epoch(loader)
    assert loss > 0
