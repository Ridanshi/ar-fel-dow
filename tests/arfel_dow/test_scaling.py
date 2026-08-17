import numpy as np
from arfel_dow.scaling import FeatureScaler


def test_transform_produces_zero_mean_unit_variance():
    X = np.array([[10.0, 1000.0], [20.0, 2000.0], [30.0, 3000.0], [40.0, 4000.0]])
    scaler = FeatureScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-6)


def test_inverse_transform_recovers_original():
    X = np.array([[10.0, 1000.0], [20.0, 2000.0], [30.0, 3000.0], [40.0, 4000.0]])
    scaler = FeatureScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)
    assert np.allclose(X_recovered, X, atol=1e-6)


def test_transform_on_new_data_uses_fitted_stats_not_new_data_stats():
    X_fit = np.array([[10.0], [20.0], [30.0], [40.0]])
    scaler = FeatureScaler().fit(X_fit)
    X_new = np.array([[100.0], [200.0]])
    X_new_scaled = scaler.transform(X_new)
    # Using X_fit's mean(25)/std, not X_new's own mean/std.
    expected = (X_new - X_fit.mean(axis=0)) / (X_fit.std(axis=0) + 1e-8)
    assert np.allclose(X_new_scaled, expected, atol=1e-6)


def test_handles_zero_variance_feature_without_dividing_by_zero():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])  # first column is constant
    scaler = FeatureScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert np.all(np.isfinite(X_scaled))
