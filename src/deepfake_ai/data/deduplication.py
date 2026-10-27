"""Perceptual deduplication of video frames and identical media."""

import pandas as pd


def deduplicate_media_samples(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=['sample_id']).copy()
