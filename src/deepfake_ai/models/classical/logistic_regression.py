"""Regularized Logistic Regression baseline."""

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000, class_weight='balanced', random_state=42)
