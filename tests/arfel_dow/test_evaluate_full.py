import numpy as np
from arfel_dow.evaluate import run_full_evaluation

RNG = np.random.default_rng(42)

FEATURE_NAMES = [
    "SubmitTime",
    "InvocationDelay",
    "ResponseDelay",
    "FunctionDuration",
    "ActiveFunctionsAtRequest",
    "ActiveFunctionsAtResponse",
]


def _toy_separable_data(n=300):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    # Shuffle so contiguous client partitions (see _partition_for_clients in
    # evaluate.py) each contain both classes, matching test_federated.py's
    # fixture convention.
    perm = RNG.permutation(n)
    X, y = X[perm], y[perm]
    durations_ms = np.where(y == 1, 5000.0, 500.0)
    memory_mb = np.full(n, 512.0)
    return X, y, durations_ms, memory_mb


def test_run_full_evaluation_returns_base_keys_plus_federated_and_narrative():
    X, y, durations_ms, memory_mb = _toy_separable_data()
    results = run_full_evaluation(
        X, y, durations_ms, memory_mb, epsilon=2.0, feature_names=FEATURE_NAMES, n_clients=3
    )

    base_keys = {
        "clean_accuracy",
        "adversarial_accuracy_plain",
        "adversarial_accuracy_robust",
        "baseline_clean_accuracy",
        "baseline_adversarial_accuracy",
        "financial_robustness_gap_plain",
        "financial_robustness_gap_robust",
    }
    extended_keys = {
        "federated_clean_accuracy",
        "federated_adversarial_accuracy",
        "sample_narrative",
    }
    assert base_keys.issubset(results.keys())
    assert extended_keys.issubset(results.keys())

    for key in base_keys | {"federated_clean_accuracy", "federated_adversarial_accuracy"}:
        assert isinstance(results[key], float)
    assert isinstance(results["sample_narrative"], str)
    assert len(results["sample_narrative"]) > 0


def test_run_full_evaluation_federated_clean_accuracy_is_high_on_separable_data():
    X, y, durations_ms, memory_mb = _toy_separable_data()
    results = run_full_evaluation(
        X, y, durations_ms, memory_mb, epsilon=2.0, feature_names=FEATURE_NAMES, n_clients=3
    )
    assert results["federated_clean_accuracy"] > 0.9


def test_sample_narrative_mentions_a_real_feature_name():
    X, y, durations_ms, memory_mb = _toy_separable_data()
    results = run_full_evaluation(
        X, y, durations_ms, memory_mb, epsilon=2.0, feature_names=FEATURE_NAMES, n_clients=3
    )
    assert any(name in results["sample_narrative"] for name in FEATURE_NAMES)
