import numpy as np
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.federated import FederatedEnsembleTrainer

RNG = np.random.default_rng(42)


def _toy_separable_data(n=300):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    perm = RNG.permutation(n)
    return X[perm], y[perm]


def _three_client_partitions(X, y, n_per_client=100):
    return [
        (X[0:n_per_client], y[0:n_per_client]),
        (X[n_per_client:2 * n_per_client], y[n_per_client:2 * n_per_client]),
        (X[2 * n_per_client:3 * n_per_client], y[2 * n_per_client:3 * n_per_client]),
    ]


def test_fit_predict_shapes():
    X, y = _toy_separable_data()
    partitions = _three_client_partitions(X, y)
    trainer = FederatedEnsembleTrainer().fit(partitions, random_state=42)

    preds = trainer.predict(X)
    proba = trainer.predict_proba(X)

    assert preds.shape == (len(X),)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_clean_federated_accuracy_is_high():
    X, y = _toy_separable_data()
    partitions = _three_client_partitions(X, y)
    trainer = FederatedEnsembleTrainer().fit(partitions, random_state=42)
    accuracy = (trainer.predict(X) == y).mean()
    assert accuracy > 0.9


def test_federated_aggregation_dilutes_one_poisoned_client_better_than_centralized_pooling():
    X, y = _toy_separable_data()
    partitions = _three_client_partitions(X, y)

    # Poison client 0's labels: flip 60% of them.
    poisoned_X, poisoned_y = partitions[0]
    poisoned_y = poisoned_y.copy()
    flip_mask = RNG.random(len(poisoned_y)) < 0.6
    poisoned_y[flip_mask] = 1 - poisoned_y[flip_mask]
    poisoned_partitions = [(poisoned_X, poisoned_y), partitions[1], partitions[2]]

    # Federated: poisoned client is one of three independently-trained voters.
    federated_trainer = FederatedEnsembleTrainer().fit(poisoned_partitions, random_state=42)
    federated_accuracy = (federated_trainer.predict(X) == y).mean()

    # Centralized baseline: same poisoned rows mixed directly into one pooled
    # training set for a single StackedEnsembleDetector.
    pooled_X = np.vstack([poisoned_X, partitions[1][0], partitions[2][0]])
    pooled_y = np.concatenate([poisoned_y, partitions[1][1], partitions[2][1]])
    centralized = StackedEnsembleDetector(random_state=42).fit(pooled_X, pooled_y)
    centralized_accuracy = (centralized.predict(X) == y).mean()

    assert federated_accuracy >= centralized_accuracy
