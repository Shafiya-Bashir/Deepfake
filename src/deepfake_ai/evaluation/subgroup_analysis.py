"""Performance auditing across compression levels and manipulation types."""

import pandas as pd
from typing import Dict, Any


def analyze_subgroups(df: pd.DataFrame, group_col: str = 'manipulation_type') -> pd.DataFrame:
    return df.groupby(group_col).agg({'label': 'count'}).rename(columns={'label': 'count'})
