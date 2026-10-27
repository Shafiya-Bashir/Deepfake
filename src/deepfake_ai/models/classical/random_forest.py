"""Balanced Random Forest on handcrafted forensic descriptors."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(n_estimators: int = 200, max_depth: int = 20) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
