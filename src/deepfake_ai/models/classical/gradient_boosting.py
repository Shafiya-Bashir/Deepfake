"""Gradient Boosted Decision Trees."""

from sklearn.ensemble import GradientBoostingClassifier


def build_gradient_boosting(n_estimators: int = 300, learning_rate: float = 0.05) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=6,
        random_state=42
    )
