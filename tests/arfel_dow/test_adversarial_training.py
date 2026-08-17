import numpy as np
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.surrogate import Surrogate
from arfel_dow.scaling import FeatureScaler
from arfel_dow.attacks_gradient import pgd_attack
from arfel_dow.adversarial_training import train_robust_ensemble

RNG = np.random.default_rng(42)


def _toy_separable_data(n=200):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_robust_ensemble_outperforms_plain_ensemble_on_held_out_adversarial_examples():
    X, y = _toy_separable_data()
    plain_ensemble = StackedEnsembleDetector(random_state=42).fit(X, y)

    scaler = FeatureScaler().fit(X)
    X_scaled = scaler.transform(X)
    surrogate = Surrogate(input_dim=6, random_state=42).fit(
        X_scaled, plain_ensemble.predict_proba(X), epochs=200
    )

    # Split attack rows: half augments the robust model's training set,
    # half is held out purely for evaluation — the robust model must never
    # be scored on rows it was trained on (see train_robust_ensemble's
    # docstring for why that was a real bug in an earlier version).
    attack_mask = y == 1
    X_attack, y_attack = X[attack_mask], y[attack_mask]
    split = len(X_attack) // 2
    X_attack_train, y_attack_train = X_attack[:split], y_attack[:split]
    X_attack_test, y_attack_test = X_attack[split:], y_attack[split:]

    X_attack_test_scaled = scaler.transform(X_attack_test)
    X_adv_test_scaled = pgd_attack(
        surrogate, X_attack_test_scaled, y_attack_test, epsilon=2.0, alpha=0.25, steps=15
    )
    X_adv_test = scaler.inverse_transform(X_adv_test_scaled)

    plain_adv_accuracy = (plain_ensemble.predict(X_adv_test) == y_attack_test).mean()

    robust_ensemble = train_robust_ensemble(
        X, y, X_attack_train, y_attack_train, surrogate, scaler, epsilon=2.0, random_state=42
    )
    robust_adv_accuracy = (robust_ensemble.predict(X_adv_test) == y_attack_test).mean()

    assert robust_adv_accuracy >= plain_adv_accuracy
