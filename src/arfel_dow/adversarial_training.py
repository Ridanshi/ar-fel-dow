import numpy as np
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.surrogate import Surrogate
from arfel_dow.scaling import FeatureScaler
from arfel_dow.attacks_gradient import pgd_attack
from arfel_dow.weighting import balanced_sample_weight


def train_robust_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    X_attack_augment: np.ndarray,
    y_attack_augment: np.ndarray,
    surrogate: Surrogate,
    scaler: FeatureScaler,
    epsilon: float,
    random_state: int = 42,
) -> StackedEnsembleDetector:
    """Augment the training set with PGD-perturbed copies of
    (X_attack_augment, y_attack_augment), then retrain the ensemble on
    clean + adversarial data — standard adversarial training, matching the
    source AR-FEL paper's defense procedure.

    X_attack_augment/y_attack_augment must be a subset of attack-class rows
    DISJOINT from whatever rows the caller later evaluates the returned
    model's adversarial robustness against. An earlier version of this
    function derived the attack rows internally from the full (X, y),
    which meant the exact same adversarial examples used to train the
    'robust' model were also used to evaluate it afterward — the model was
    scoring on memorized training rows, not held-out ones, producing an
    unrealistic near-perfect accuracy that masked whether adversarial
    training generalizes at all. Callers now split attack rows into a
    train subset (passed here) and a test subset (evaluated separately)
    before calling this function.

    `surrogate` must have been fit on scaler.transform(X) (standardized
    space) — see scaling.py's docstring for why: epsilon is meaningless as
    a single raw-unit constant across heterogeneous real feature scales.
    `epsilon` is therefore in standardized (zero-mean, unit-variance)
    units, not raw feature units."""
    X_attack_scaled = scaler.transform(X_attack_augment)
    X_adv_scaled = pgd_attack(
        surrogate, X_attack_scaled, y_attack_augment, epsilon=epsilon, alpha=epsilon / 8, steps=15
    )
    X_adv = scaler.inverse_transform(X_adv_scaled)

    X_augmented = np.vstack([X, X_adv])
    y_augmented = np.concatenate([y, y_attack_augment])

    robust_ensemble = StackedEnsembleDetector(random_state=random_state)
    robust_ensemble.fit(X_augmented, y_augmented, sample_weight=balanced_sample_weight(y_augmented))
    return robust_ensemble
