"""Federated poisoning check requested during review (Section 4.6's
anti-poisoning claim): the paper states a malicious tenant's poisoned
labels dilute to "roughly 1/N of the aggregate vote" but never actually
ran a poisoning attack to verify it — an unverified claim dressed as a
finding. This runs one.

Threat model: one of three tenants is malicious and label-flips all of
its own attack-class rows to benign ("hide the attack" poisoning — the
most damaging goal for this specific application, since it teaches the
shared model to miss real attacks). Compares three conditions on the
same held-out test set:
  1. federated, unpoisoned (control — should match the headline ~99%)
  2. federated, one tenant poisoned
  3. centralized, same total data including the poisoned rows, pooled
     into one training set (no per-tenant isolation at all)

If federation's isolation is doing anything, (2) should be measurably
less degraded than (3) on attack_recall specifically.

Scoped to a 50k-row sample, single seed — a targeted check, not a repeat
of the 5-seed headline run.
"""
import json
import numpy as np

from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.evaluate import _train_test_split, _partition_for_clients
from arfel_dow.federated import FederatedEnsembleTrainer
from arfel_dow.weighting import balanced_sample_weight
from run_experiment import load_and_prepare

SEED = 42
SAMPLE_SIZE = 50_000
N_CLIENTS = 3


def evaluate(y_true, y_pred, label):
    accuracy = float((y_pred == y_true).mean())
    attack_recall = float((y_pred[y_true == 1] == 1).mean())
    benign_recall = float((y_pred[y_true == 0] == 0).mean())
    row = {"condition": label, "accuracy": accuracy, "attack_recall": attack_recall, "benign_recall": benign_recall}
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    X, y, durations_ms, memory_mb, groups, feature_columns = load_and_prepare(sample_size=SAMPLE_SIZE)
    X_train, y_train, X_test, y_test, _, _ = _train_test_split(
        X, y, test_size=0.3, random_state=SEED, groups=groups
    )

    clean_partitions = _partition_for_clients(X_train, y_train, N_CLIENTS, random_state=SEED)

    # Poison client 0: flip 90% of its attack-class labels to benign
    # (randomly chosen). Not 100% — a full flip leaves that shard
    # single-class, which crashes CatBoost's fit outright rather than
    # producing a degraded-but-valid model; 90% is still a severe,
    # realistic "hide almost all attacks" poisoning goal while leaving
    # enough of both classes for every base learner to fit. Clients 1
    # and 2 are untouched.
    poison_rng = np.random.default_rng(SEED)
    poisoned_partitions = []
    for i, (X_c, y_c) in enumerate(clean_partitions):
        if i == 0:
            y_c = y_c.copy()
            attack_idx = np.flatnonzero(y_c == 1)
            n_flip = int(round(len(attack_idx) * 0.9))
            flip_idx = poison_rng.choice(attack_idx, size=n_flip, replace=False)
            y_c[flip_idx] = 0
        poisoned_partitions.append((X_c, y_c))

    n_attack_before = int((clean_partitions[0][1] == 1).sum())
    n_attack_after = int((poisoned_partitions[0][1] == 1).sum())
    print(f"client 0: {n_attack_before} attack rows before poisoning, {n_attack_after} remain "
          f"after flipping 90% to benign ({n_attack_before - n_attack_after} flipped)", flush=True)

    results = []

    print("=== condition 1: federated, unpoisoned (control) ===", flush=True)
    fed_clean = FederatedEnsembleTrainer().fit(clean_partitions, random_state=SEED)
    results.append(evaluate(y_test, fed_clean.predict(X_test), "federated_unpoisoned"))

    print("=== condition 2: federated, one tenant poisoned ===", flush=True)
    fed_poisoned = FederatedEnsembleTrainer().fit(poisoned_partitions, random_state=SEED)
    results.append(evaluate(y_test, fed_poisoned.predict(X_test), "federated_poisoned"))

    print("=== condition 3: centralized, same poisoned data pooled ===", flush=True)
    X_pooled = np.vstack([X_c for X_c, _ in poisoned_partitions])
    y_pooled = np.concatenate([y_c for _, y_c in poisoned_partitions])
    centralized_poisoned = StackedEnsembleDetector(random_state=SEED).fit(
        X_pooled, y_pooled, sample_weight=balanced_sample_weight(y_pooled)
    )
    results.append(evaluate(y_test, centralized_poisoned.predict(X_test), "centralized_poisoned"))

    print("\n=== SUMMARY (attack_recall is the metric that matters for this threat model) ===", flush=True)
    for r in results:
        print(f"{r['condition']}: attack_recall={r['attack_recall']:.4f}, accuracy={r['accuracy']:.4f}", flush=True)

    with open("results_federated_poisoning.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results_federated_poisoning.json", flush=True)
