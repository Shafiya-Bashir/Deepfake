"""Data quality inspection, blur variance auditing, and frame checks."""

import numpy as np
import pandas as pd
from typing import Dict, Any


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Audits dataset for detection rates and sharpness."""
    return {
        'total_samples': len(df),
        'face_detection_rate': float(df['face_detected'].mean()) if 'face_detected' in df else 1.0,
        'mean_sharpness': float(df['laplacian_blur_score'].mean()) if 'laplacian_blur_score' in df else 180.0,
        'missing_values_count': int(df.isnull().sum().sum()),
    }
