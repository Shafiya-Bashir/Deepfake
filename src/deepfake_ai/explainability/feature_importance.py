"""Feature importance calculation for classical forensic models."""

import pandas as pd
import numpy as np


def extract_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
