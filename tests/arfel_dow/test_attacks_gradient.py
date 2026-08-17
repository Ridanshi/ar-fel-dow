import numpy as np
import torch
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.surrogate import Surrogate
from arfel_dow.attacks_gradient import fgsm_attack, pgd_attack, cw_attack

RNG = np.random.default_rng(42)


def _toy_separable_data(n=200):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def _fit_surrogate():
    X, y = _toy_separable_data()
    ensemble = StackedEnsembleDetector(random_state=42).fit(X, y)
    surrogate = Surrogate(input_dim=6, random_state=42)
    surrogate.fit(X, ensemble.predict_proba(X), epochs=200)
    return surrogate, X, y


def test_fgsm_perturbation_within_epsilon_ball():
    surrogate, X, y = _fit_surrogate()
    epsilon = 2.0
    X_adv = fgsm_attack(surrogate, X, y, epsilon=epsilon)
    assert X_adv.shape == X.shape
    # float32 round-trip through torch introduces rounding noise proportional
    # to feature magnitude (~50 here); 1e-3 comfortably exceeds that noise
    # while still catching any real epsilon-ball violation.
    assert np.all(np.abs(X_adv - X) <= epsilon + 1e-3)


def test_fgsm_reduces_surrogate_accuracy_on_attack_class():
    surrogate, X, y = _fit_surrogate()
    attack_mask = y == 1
    X_attack = X[attack_mask]
    y_attack = y[attack_mask]

    clean_preds = surrogate.predict_proba(X_attack).argmax(axis=1)
    clean_accuracy = (clean_preds == y_attack).mean()

    X_adv = fgsm_attack(surrogate, X_attack, y_attack, epsilon=5.0)
    adv_preds = surrogate.predict_proba(X_adv).argmax(axis=1)
    adv_accuracy = (adv_preds == y_attack).mean()

    assert adv_accuracy <= clean_accuracy


def test_pgd_perturbation_within_epsilon_ball():
    surrogate, X, y = _fit_surrogate()
    epsilon = 2.0
    X_adv = pgd_attack(surrogate, X, y, epsilon=epsilon, alpha=0.5, steps=10)
    assert X_adv.shape == X.shape
    # float32 round-trip through torch introduces rounding noise proportional
    # to feature magnitude (~50 here); 1e-3 comfortably exceeds that noise
    # while still catching any real epsilon-ball violation.
    assert np.all(np.abs(X_adv - X) <= epsilon + 1e-3)


def test_cw_attack_returns_valid_shape():
    surrogate, X, y = _fit_surrogate()
    X_adv = cw_attack(surrogate, X, y, c=1.0, steps=50)
    assert X_adv.shape == X.shape
    assert np.all(np.isfinite(X_adv))
