"""Actor-level K-Fold cross validation partitioner."""

from sklearn.model_selection import KFold


def get_kfold_splits(n_splits: int = 5):
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)
