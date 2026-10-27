"""Identity-disjoint train/validation/test splitting."""

import pandas as pd
from typing import Tuple
from deepfake_ai.utils.hashing import compute_md5


def split_identity_disjoint(
    df: pd.DataFrame,
    identity_col: str = 'actor_id',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset ensuring actor IDs are strictly disjoint between sets."""
    unique_actors = df[identity_col].unique()

    def actor_to_bucket(actor: str) -> int:
        return int(compute_md5(str(actor))[:4], 16) % 100

    train_thresh = int(train_ratio * 100)
    val_thresh = int((train_ratio + val_ratio) * 100)

    train_actors = [a for a in unique_actors if actor_to_bucket(a) < train_thresh]
    val_actors = [a for a in unique_actors if train_thresh <= actor_to_bucket(a) < val_thresh]
    test_actors = [a for a in unique_actors if actor_to_bucket(a) >= val_thresh]

    train_df = df[df[identity_col].isin(train_actors)].copy()
    val_df = df[df[identity_col].isin(val_actors)].copy()
    test_df = df[df[identity_col].isin(test_actors)].copy()

    return train_df, val_df, test_df
