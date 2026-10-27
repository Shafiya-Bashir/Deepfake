"""Cross-model performance matrix compiler."""

import pandas as pd
from typing import List, Dict, Any


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if 'roc_auc' in df.columns:
        df = df.sort_values(by='roc_auc', ascending=False)
    return df
