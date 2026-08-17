"""Two generalization checks requested during review (Section 6.2/8.2's
cold-start caveat and the IP-identity-construct concern in Section 4.1):

(a) Cold-start slice analysis: does the detector's attack recall actually
    degrade for attacking IPs early in their request history (low
    ip_request_sequence_number), where the dominant engineered feature
    carries the least signal? Buckets the held-out test set by that raw
    feature value and reports per-bucket accuracy/recall.

(b) Temporal split: the headline pipeline splits train/test by source IP
    (GroupShuffleSplit) to prevent identity leakage, but every row from a
    given IP — early or late in that IP's history — can still land in
    train. This reruns clean evaluation with a time-based split instead
    (earliest 70% of requests by timestamp train, latest 30% test), which
    additionally holds out *future* IPs the model never saw during
    training at all, a stricter and more production-realistic test.

Scoped to a 50k-row sample (not the full 187k dataset, single seed) since
this is a targeted robustness check, not a repeat of the 5-seed headline
run — the sample still preserves the natural class ratio (see
run_experiment.load_and_prepare).
"""
import json
import numpy as np
import pandas as pd

from arfel_dow.data import load_dataset
from arfel_dow.features import extract_features
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.evaluate import _train_test_split
from arfel_dow.weighting import balanced_sample_weight
from run_experiment import (
    load_and_prepare, DATA_PATH, LABEL_COLUMN, POSITIVE_LABEL,
    CORE_FEATURE_COLUMNS, EXTRA_NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    IP_FREQUENCY_COLUMN,
)

SEED = 42
SAMPLE_SIZE = 50_000
BUCKET_EDGES = [0, 1, 3, 10, 50, float("inf")]
BUCKET_LABELS = ["0", "1-2", "3-9", "10-49", "50+"]


def bucket_report(sequence_numbers, y_true, y_pred):
    rows = []
    for lo, hi, label in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], BUCKET_LABELS):
        mask = (sequence_numbers >= lo) & (sequence_numbers < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": label, "n": 0, "accuracy": None, "attack_recall": None})
            continue
        acc = float((y_pred[mask] == y_true[mask]).mean())
        attack_mask = mask & (y_true == 1)
        n_attack = int(attack_mask.sum())
        attack_recall = float((y_pred[attack_mask] == 1).mean()) if n_attack > 0 else None
        rows.append({"bucket": label, "n": n, "n_attack": n_attack, "accuracy": acc, "attack_recall": attack_recall})
    return rows


def coldstart_check():
    print("=== (a) Cold-start bucket analysis (group-by-IP split, seed 42) ===", flush=True)
    X, y, durations_ms, memory_mb, groups, feature_columns = load_and_prepare(sample_size=SAMPLE_SIZE)
    ip_freq_idx = feature_columns.index(IP_FREQUENCY_COLUMN)

    X_train, y_train, X_test, y_test, _, _ = _train_test_split(
        X, y, test_size=0.3, random_state=SEED, groups=groups
    )
    model = StackedEnsembleDetector(random_state=SEED).fit(
        X_train, y_train, sample_weight=balanced_sample_weight(y_train)
    )
    y_pred = model.predict(X_test)
    seq_numbers_test = X_test[:, ip_freq_idx]

    rows = bucket_report(seq_numbers_test, y_test, y_pred)
    for r in rows:
        print(json.dumps(r), flush=True)
    return rows


def temporal_split_check():
    print("=== (b) Temporal split (earliest 70% train / latest 30% test, seed 42) ===", flush=True)
    df = load_dataset(DATA_PATH, label_column=LABEL_COLUMN)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)
    dummy_columns = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_COLUMNS)]

    df["_timestamp_parsed"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(["IP", "_timestamp_parsed"]).reset_index(drop=True)
    df[IP_FREQUENCY_COLUMN] = df.groupby("IP").cumcount()

    # Now re-sort by timestamp alone (not IP-then-timestamp) so a
    # contiguous time-ordered split is meaningful; the causal IP-frequency
    # feature computed above is unaffected by this second sort since it's
    # already a fixed per-row value.
    df = df.sort_values("_timestamp_parsed").reset_index(drop=True)
    df = df.drop(columns=["_timestamp_parsed"])

    feature_columns = CORE_FEATURE_COLUMNS + EXTRA_NUMERIC_COLUMNS + [IP_FREQUENCY_COLUMN] + dummy_columns

    rng = np.random.default_rng(SEED)
    if len(df) > SAMPLE_SIZE:
        # Sample while preserving time order: pick a random subset of row
        # positions, then keep them in their original (time-sorted) order,
        # rather than df.sample()'s unordered draw, which would destroy
        # the temporal contiguity this check depends on.
        keep_idx = np.sort(rng.choice(len(df), size=SAMPLE_SIZE, replace=False))
        df = df.iloc[keep_idx].reset_index(drop=True)

    X, y = extract_features(df, feature_columns, label_column=LABEL_COLUMN, positive_label=POSITIVE_LABEL)

    split = int(len(X) * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    train_ips = set(df["IP"].iloc[:split])
    test_ips = set(df["IP"].iloc[split:])
    unseen_ip_frac = len(test_ips - train_ips) / max(len(test_ips), 1)

    print(f"train_attack_frac={y_train.mean():.4f} test_attack_frac={y_test.mean():.4f} "
          f"unseen_test_ip_fraction={unseen_ip_frac:.4f}", flush=True)

    if y_train.min() == y_train.max() or y_test.min() == y_test.max():
        print("DEGENERATE SPLIT: train or test partition is single-class after temporal "
              "ordering — attack traffic is not spread evenly across the dataset's time "
              "span. Reporting this as the finding rather than forcing a metric.", flush=True)
        return {"degenerate": True, "train_attack_frac": float(y_train.mean()), "test_attack_frac": float(y_test.mean())}

    model = StackedEnsembleDetector(random_state=SEED).fit(
        X_train, y_train, sample_weight=balanced_sample_weight(y_train)
    )
    y_pred = model.predict(X_test)
    accuracy = float((y_pred == y_test).mean())
    attack_recall = float((y_pred[y_test == 1] == 1).mean())
    benign_recall = float((y_pred[y_test == 0] == 0).mean())

    result = {
        "degenerate": False,
        "accuracy": accuracy,
        "attack_recall": attack_recall,
        "benign_recall": benign_recall,
        "unseen_test_ip_fraction": unseen_ip_frac,
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    coldstart_rows = coldstart_check()
    temporal_result = temporal_split_check()
    with open("results_generalization_checks.json", "w") as f:
        json.dump({"coldstart_buckets": coldstart_rows, "temporal_split": temporal_result}, f, indent=2)
    print("\nwrote results_generalization_checks.json", flush=True)
