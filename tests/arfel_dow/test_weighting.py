import numpy as np
from arfel_dow.weighting import balanced_sample_weight


def test_minority_class_gets_higher_weight():
    y = np.array([0] * 90 + [1] * 10)
    weights = balanced_sample_weight(y)
    assert weights[y == 1].mean() > weights[y == 0].mean()


def test_weighted_class_totals_are_balanced():
    y = np.array([0] * 90 + [1] * 10)
    weights = balanced_sample_weight(y)
    assert np.isclose(weights[y == 0].sum(), weights[y == 1].sum())
