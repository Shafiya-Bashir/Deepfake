import pytest
import torch
import pandas as pd
from pathlib import Path
from deepfake_ai.data.streaming_dataset import GoogleDriveStreamingDataset, KaggleDriveStreamer


@pytest.mark.unit
def test_google_drive_streaming_dataset_init():
    df = pd.DataFrame([
        {'sample_id': '001', 'label': 0, 'dataset': 'DFDC', 'actor_id': 'act_01', 'manipulation_type': 'real'},
        {'sample_id': '002', 'label': 1, 'dataset': 'DFDC', 'actor_id': 'act_02', 'manipulation_type': 'deepfake'}
    ])
    ds = GoogleDriveStreamingDataset(
        metadata_csv_or_df=df,
        cache_in_memory=True
    )
    assert len(ds) == 2
    item = ds[0]
    assert "image" in item
    assert "label" in item
    assert item["image"].shape == (3, 224, 224)
    assert item["label"].item() == 0


@pytest.mark.unit
def test_kaggle_drive_streamer():
    streamer = KaggleDriveStreamer()
    assert hasattr(streamer, "setup_kaggle_credentials")
    assert hasattr(streamer, "download_competition_chunk_to_drive")
