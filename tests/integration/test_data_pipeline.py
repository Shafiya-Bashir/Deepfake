import pytest
import pandas as pd
from deepfake_ai.data.data_splitting import split_identity_disjoint


@pytest.mark.integration
def test_split_pipeline():
    df = pd.DataFrame({
        'sample_id': [f's_{i}' for i in range(20)],
        'actor_id': [f'act_{i % 5}' for i in range(20)],
        'label': [i % 2 for i in range(20)]
    })
    tr, val, ts = split_identity_disjoint(df)
    assert len(tr) > 0
    assert len(val) >= 0
    assert len(ts) >= 0
