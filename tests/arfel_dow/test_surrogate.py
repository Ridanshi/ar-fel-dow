import numpy as np
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.surrogate import Surrogate

RNG = np.random.default_rng(42)


def _toy_separable_data(n=200):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_surrogate_mimics_ensemble_predictions():
    X, y = _toy_separable_data()
    ensemble = StackedEnsembleDetector(random_state=42).fit(X, y)
    ensemble_proba = ensemble.predict_proba(X)

    surrogate = Surrogate(input_dim=6, random_state=42)
    surrogate.fit(X, ensemble_proba, epochs=200)
    surrogate_proba = surrogate.predict_proba(X)

    surrogate_labels = surrogate_proba.argmax(axis=1)
    ensemble_labels = ensemble_proba.argmax(axis=1)
    agreement = (surrogate_labels == ensemble_labels).mean()
    assert agreement > 0.9


def test_surrogate_forward_is_differentiable():
    import torch

    surrogate = Surrogate(input_dim=6, random_state=42)
    x = torch.randn(4, 6, requires_grad=True)
    out = surrogate.forward(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 6)
