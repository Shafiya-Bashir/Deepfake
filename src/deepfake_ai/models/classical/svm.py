"""Support Vector Classifier with RBF kernel and probability calibration."""

from sklearn.svm import SVC


def build_svm(C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale') -> SVC:
    return SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
