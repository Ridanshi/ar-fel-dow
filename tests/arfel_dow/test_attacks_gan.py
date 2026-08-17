import numpy as np
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.surrogate import Surrogate
from arfel_dow.attacks_gan import AdaptiveGANAttacker

RNG = np.random.default_rng(42)


def _toy_separable_data(n=200):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y, benign, attack


def test_generate_returns_correct_shape():
    X, y, benign, attack = _toy_separable_data()
    ensemble = StackedEnsembleDetector(random_state=42).fit(X, y)
    surrogate = Surrogate(input_dim=6, random_state=42).fit(X, ensemble.predict_proba(X), epochs=200)

    gan = AdaptiveGANAttacker(input_dim=6, noise_dim=8, random_state=42)
    gan.fit(X_benign=benign, X_attack=attack, surrogate=surrogate, epochs=50)

    generated = gan.generate(n_samples=20)
    assert generated.shape == (20, 6)
    assert np.all(np.isfinite(generated))


def test_generated_samples_evade_more_than_random_noise_baseline():
    X, y, benign, attack = _toy_separable_data()
    ensemble = StackedEnsembleDetector(random_state=42).fit(X, y)
    surrogate = Surrogate(input_dim=6, random_state=42).fit(X, ensemble.predict_proba(X), epochs=200)

    gan = AdaptiveGANAttacker(input_dim=6, noise_dim=8, random_state=42)
    gan.fit(X_benign=benign, X_attack=attack, surrogate=surrogate, epochs=100)
    generated = gan.generate(n_samples=50)
    generated_evasion_rate = (surrogate.predict_proba(generated).argmax(axis=1) == 0).mean()

    rng = np.random.default_rng(1)
    random_noise = rng.normal(loc=attack.mean(axis=0), scale=attack.std(axis=0), size=(50, 6))
    random_evasion_rate = (surrogate.predict_proba(random_noise).argmax(axis=1) == 0).mean()

    assert generated_evasion_rate >= random_evasion_rate
