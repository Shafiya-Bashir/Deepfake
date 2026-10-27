"""Stacking meta-learner combining multiple model predictions."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class StackingEnsemble:
    def __init__(self):
        self.meta_classifier = LogisticRegression()

    def fit(self, base_predictions: np.ndarray, y_true: np.ndarray):
        self.meta_classifier.fit(base_predictions, y_true)

    def predict_proba(self, base_predictions: np.ndarray) -> np.ndarray:
        return self.meta_classifier.predict_proba(base_predictions)
