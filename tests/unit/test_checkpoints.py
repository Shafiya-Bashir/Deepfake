import torch
import pytest
from pathlib import Path
from deepfake_ai.training.checkpoint_manager import CheckpointManager
from deepfake_ai.models.deep_learning.mesonet import Meso4


@pytest.mark.unit
def test_checkpoint_save(tmp_path):
    ckpt_mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
    model = Meso4()
    opt = torch.optim.Adam(model.parameters())
    ckpt_mgr.save_checkpoint(model, opt, epoch=1, metrics={'auc': 0.95}, is_best=True, model_name='test_meso')
    assert (tmp_path / 'best/test_meso_best.pt').exists()
