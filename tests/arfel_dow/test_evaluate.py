import numpy as np
from arfel_dow.evaluate import run_evaluation, _train_test_split

RNG = np.random.default_rng(42)


def _toy_separable_data(n=200):
    benign = RNG.normal(loc=[10, 10, 10, 10, 10, 10], scale=1.0, size=(n // 2, 6))
    attack = RNG.normal(loc=[50, 50, 50, 50, 50, 50], scale=1.0, size=(n // 2, 6))
    X = np.vstack([benign, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    durations_ms = np.where(y == 1, 5000.0, 500.0)
    memory_mb = np.full(n, 512.0)
    return X, y, durations_ms, memory_mb


def test_train_test_split_keeps_groups_disjoint_when_groups_given():
    # Real dataset finding: a single IP can contribute hundreds of
    # near-identical rows. Row-level shuffling scatters the same IP's rows
    # across both train and test, letting the model partially recognize an
    # IP it already saw during training — a leakage source distinct from
    # row-level train/test leakage. groups=IP forces every row from a given
    # IP onto the same side of the split.
    RNG_LOCAL = np.random.default_rng(0)
    n_groups = 20
    rows_per_group = 15
    X = RNG_LOCAL.normal(size=(n_groups * rows_per_group, 4))
    y = RNG_LOCAL.integers(0, 2, size=n_groups * rows_per_group)
    groups = np.repeat(np.arange(n_groups), rows_per_group)

    _, _, _, _, train_indices, test_indices = _train_test_split(
        X, y, test_size=0.3, random_state=42, groups=groups
    )
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])
    assert train_groups.isdisjoint(test_groups)


def test_train_test_split_ignores_groups_when_none():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.array([0, 1] * 10)
    X_train, y_train, X_test, y_test, train_indices, test_indices = _train_test_split(
        X, y, test_size=0.3, random_state=42, groups=None
    )
    assert len(train_indices) + len(test_indices) == 20
    assert set(train_indices.tolist()).isdisjoint(set(test_indices.tolist()))


def test_run_evaluation_returns_expected_keys_and_types():
    X, y, durations_ms, memory_mb = _toy_separable_data()
    results = run_evaluation(X, y, durations_ms, memory_mb, epsilon=2.0)

    expected_keys = {
        "clean_accuracy",
        "clean_attack_recall",
        "clean_benign_recall",
        "adversarial_accuracy_plain",
        "adversarial_accuracy_robust",
        "adversarial_accuracy_plain_gan",
        "adversarial_accuracy_robust_gan",
        "baseline_clean_accuracy",
        "baseline_adversarial_accuracy",
        "financial_robustness_gap_plain",
        "financial_robustness_gap_robust",
    }
    assert set(results.keys()) == expected_keys
    for key in expected_keys:
        assert isinstance(results[key], float)
    accuracy_keys = {k for k in expected_keys if "gap" not in k}
    for key in accuracy_keys:
        assert 0.0 <= results[key] <= 1.0


def test_per_class_recall_is_high_on_trivially_separable_data():
    # Blended accuracy can mask a model that's much better at one class than
    # the other (exactly what happened on the real DoWTS dataset: good
    # benign recall, poor attack recall, masked by a single accuracy
    # number). On trivially separable toy data both per-class recalls
    # should be high, confirming the two figures are computed correctly
    # and independently of each other.
    X, y, durations_ms, memory_mb = _toy_separable_data()
    results = run_evaluation(X, y, durations_ms, memory_mb, epsilon=2.0)
    assert results["clean_attack_recall"] > 0.9
    assert results["clean_benign_recall"] > 0.9


def test_robust_accuracy_under_attack_is_at_least_plain_accuracy_under_attack():
    X, y, durations_ms, memory_mb = _toy_separable_data()
    results = run_evaluation(X, y, durations_ms, memory_mb, epsilon=2.0)
    assert results["adversarial_accuracy_robust"] >= results["adversarial_accuracy_plain"]


def test_gan_accuracies_are_valid_probabilities():
    # PGD-adversarial training may or may not generalize to a structurally
    # different attack (GAN) — that's an open empirical question this
    # metric answers, not something to assert a direction for. Only check
    # the numbers are well-formed.
    X, y, durations_ms, memory_mb = _toy_separable_data()
    results = run_evaluation(X, y, durations_ms, memory_mb, epsilon=2.0)
    assert 0.0 <= results["adversarial_accuracy_plain_gan"] <= 1.0
    assert 0.0 <= results["adversarial_accuracy_robust_gan"] <= 1.0
