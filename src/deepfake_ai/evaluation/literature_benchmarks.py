"""Benchmark comparison against published SOTA literature baselines."""

LITERATURE_BASELINES = {
    'FaceForensics++_c23': {
        'MesoInception4': 0.831,
        'Xception': 0.993,
        'FrequencyCNN': 0.988,
        'Face_XRay': 0.985,
    },
    'Celeb-DF_v2': {
        'MesoInception4': 0.536,
        'Xception': 0.655,
        'LipForensics': 0.824,
        'UnivFD': 0.785,
    }
}

def compare_with_literature(dataset: str, achieved_auc: float) -> dict:
    baselines = LITERATURE_BASELINES.get(dataset, {})
    return {
        'dataset': dataset,
        'achieved_auc': achieved_auc,
        'literature_baselines': baselines,
    }
