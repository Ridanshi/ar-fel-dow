import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Per-row weights inversely proportional to class frequency, so a
    StackedEnsembleDetector can train on the full, naturally-imbalanced
    dataset instead of discarding real rows via downsampling. Confirmed
    empirically (see scripts/experiment_class_weighting.py, since removed)
    to modestly beat downsample-then-fit on the real DoWTS dataset."""
    return compute_sample_weight("balanced", y)
