import numpy as np


class FeatureScaler:
    """Standardize features to zero mean, unit variance per column.

    Exists because L-infinity-bounded gradient attacks (FGSM/PGD/C&W) treat
    a fixed epsilon as equally meaningful across every feature dimension.
    On the real DoWTS dataset, raw feature magnitudes range from single
    digits (ActiveFunctionsAtRequest) to the thousands (InvocationDelay) —
    an epsilon tuned against one feature's scale is negligible noise
    against another's, which silently made epsilon=8.0 perturb zero real
    rows enough to flip a prediction at full dataset scale. Attacking in
    standardized space makes one epsilon value comparably meaningful across
    all features."""

    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * self._std + self._mean
