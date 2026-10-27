"""Imputation and handling of missing frames and audio streams."""

import pandas as pd


def handle_missing_frames(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if 'laplacian_blur_score' in cleaned.columns:
        cleaned['laplacian_blur_score'] = cleaned['laplacian_blur_score'].fillna(cleaned['laplacian_blur_score'].median())
    return cleaned
