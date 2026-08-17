import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from arfel_dow.detector import StackedEnsembleDetector

RNG = np.random.default_rng(42)


def _toy_separable_data(n=200):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_fit_predict_on_separable_data():
    X, y = _toy_separable_data()
    detector = StackedEnsembleDetector(random_state=42)
    detector.fit(X, y)
    preds = detector.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.95


def test_predict_proba_shape_and_range():
    X, y = _toy_separable_data()
    detector = StackedEnsembleDetector(random_state=42)
    detector.fit(X, y)
    proba = detector.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_ensemble_replaces_gradient_boosting_with_catboost():
    # CatBoost replaces sklearn's GradientBoostingClassifier (a stated
    # deviation from the source AR-FEL paper's exact architecture) rather
    # than being added as a fifth learner on top of the original four.
    detector = StackedEnsembleDetector(random_state=42)
    base_learner_names = [name for name, _ in detector._model.estimators]
    assert "catboost" in base_learner_names
    assert "gradient_boosting" not in base_learner_names
    assert len(base_learner_names) == 4


def test_fit_accepts_sample_weight_for_imbalanced_training():
    # Heavily imbalanced (90 benign, 10 attack) — without weighting a
    # classifier can trivially predict the majority class and still score
    # high raw accuracy. sample_weight lets callers keep the full dataset
    # (rather than downsampling the majority class, discarding real data)
    # by up-weighting the minority class instead.
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(90, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(10, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_sample_weight("balanced", y)

    detector = StackedEnsembleDetector(random_state=42)
    detector.fit(X, y, sample_weight=weights)
    preds = detector.predict(X)
    # Classes are well-separated (loc 10 vs loc 50), so a properly-weighted
    # fit should still recover the minority class rather than collapsing
    # to predicting only "benign".
    attack_recall = (preds[y == 1] == 1).mean()
    assert attack_recall > 0.5
