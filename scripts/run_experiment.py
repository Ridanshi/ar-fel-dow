"""End-to-end AR-FEL-DoW experiment against the real DoWTS dataset.

Before running: download the dataset from
https://data.mendeley.com/datasets/g8g9vdxyvn/1 to data/raw/dowts_real.csv,
then confirm the FEATURE_COLUMNS and LABEL_COLUMN below match the real
headers recorded in docs/superpowers/plans/dataset-schema-notes.md.

Schema discovered 2026-08-12 (see dataset-schema-notes.md): the label column
is `bot` (boolean, True=attack/bot traffic), not the fixture's `label`
string column, and real memory is available per-row via `vmmemorybucket`
(GB), not a fixed assumption.
"""
import json
import pandas as pd
from arfel_dow.data import load_dataset
from arfel_dow.features import extract_features
from arfel_dow.evaluate import run_full_evaluation

DATA_PATH = "data/raw/dowts_real.csv"
LABEL_COLUMN = "bot"
POSITIVE_LABEL = True

# The source AR-FEL paper's prose describes six DoW features (the
# CORE_FEATURE_COLUMNS below). The real dataset has additional columns
# (RTT, CPU utilization percentiles, VM category, function trigger type)
# that were originally left unused to match that stated feature set —
# clean_accuracy plateaued around 63-65% regardless of ensemble tuning
# (CatBoost, hyperparameters), suggesting the six timing/concurrency
# features alone may not fully determine the `bot` label. Adding the
# dataset's remaining columns tests that directly. This is an explicit
# deviation from the paper's stated feature set, not a hidden one.
CORE_FEATURE_COLUMNS = [
    "SubmitTime",
    "InvocationDelay",
    "ResponseDelay",
    "FunctionDuration",
    "ActiveFunctionsAtRequest",
    "ActiveFunctionsAtResponse",
]
EXTRA_NUMERIC_COLUMNS = [
    "RTT",
    "maxcpu",
    "avgcpu",
    "p95maxcpu",
    "vmcorecountbucket",
    "vmmemorybucket",
]
CATEGORICAL_COLUMNS = ["vmcategory", "functionTrigger"]

# Diagnostic finding: `bot` is a pure IP-identity marker, not a behavioral
# pattern — every one of the dataset's 7,587 IPs is either 100% bot rows
# (100 IPs, ~1,310 requests each) or 0% bot rows (7,487 IPs, ~7.5 requests
# each), with zero mixed IPs. No per-request timing/resource feature can
# fully recover that without knowing the IP. IP_FREQUENCY_COLUMN below adds
# a legitimate, non-identity behavioral signal instead — how many PRIOR
# requests this same IP had made before the current one (computed causally
# per IP, sorted by timestamp) — bot IPs accumulate a high count quickly
# within the dataset's time span; legit IPs rarely exceed single digits.
# This works even for an IP never seen in the train partition, since it
# only uses that IP's own earlier rows, not any other IP's history.
IP_FREQUENCY_COLUMN = "ip_request_sequence_number"

DURATION_COLUMN = "FunctionDuration"
MEMORY_BUCKET_COLUMN = "vmmemorybucket"  # GB; convert to MB below

# Full dataset is 187,088 rows with a natural 70/30 attack/benign split.
# An earlier version of this script used balance_classes() to downsample
# the majority class before training, which discarded ~56,000 real rows.
# Controlled comparison (scripts/experiment_class_weighting.py, since
# removed) showed class-weighted training on the full imbalanced dataset
# modestly but consistently beats downsample-then-fit (higher accuracy,
# higher attack recall) — StackedEnsembleDetector now takes sample_weight
# directly (see weighting.py) and every fit call in evaluate.py/
# adversarial_training.py/federated.py applies it, so no rebalancing step
# is needed here. SAMPLE_SIZE below is now a plain random subsample (still
# preserves the natural class ratio) for faster dev-loop runs; set to None
# for the full paper-quality run, which should be launched as a longer/
# background job rather than an interactive one.
SAMPLE_SIZE = None


def load_and_prepare(sample_size: int | None = SAMPLE_SIZE, include_ip_frequency: bool = True):
    """Load DoWTS, engineer features, and return everything run_full_evaluation
    needs. Factored out of __main__ so scripts/run_multi_seed.py and the
    CatBoost-isolation ablation can reuse identical data preparation without
    duplicating it (and risking the two silently drifting apart)."""
    df = load_dataset(DATA_PATH, label_column=LABEL_COLUMN)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)
    dummy_columns = [
        c for c in df.columns
        if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_COLUMNS)
    ]

    # Causal per-IP request sequence number: sort by (IP, timestamp), then
    # count how many prior rows from the same IP precede this one. Computed
    # on the full dataset before any subsampling so counts reflect each
    # IP's real request history, not an artifact of which rows happened to
    # survive subsampling.
    df["_timestamp_parsed"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(["IP", "_timestamp_parsed"]).reset_index(drop=True)
    df[IP_FREQUENCY_COLUMN] = df.groupby("IP").cumcount()
    df = df.drop(columns=["_timestamp_parsed"])

    ip_freq_cols = [IP_FREQUENCY_COLUMN] if include_ip_frequency else []
    feature_columns = CORE_FEATURE_COLUMNS + EXTRA_NUMERIC_COLUMNS + ip_freq_cols + dummy_columns

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    X, y = extract_features(df, feature_columns, label_column=LABEL_COLUMN, positive_label=POSITIVE_LABEL)
    durations_ms = df[DURATION_COLUMN].to_numpy(dtype=float)
    memory_mb = df[MEMORY_BUCKET_COLUMN].to_numpy(dtype=float) * 1024.0
    # groups = source IP, forces every row from the same IP onto one side
    # of the train/test split (see evaluate.py's _train_test_split) — the
    # correct way to split this dataset given IP-identity-determined labels.
    groups = df["IP"].to_numpy()

    return X, y, durations_ms, memory_mb, groups, feature_columns


if __name__ == "__main__":
    X, y, durations_ms, memory_mb, groups, FEATURE_COLUMNS = load_and_prepare()

    # epsilon is in standardized (zero-mean, unit-variance) feature units,
    # not raw units — see scaling.py. 1.0 = a one-standard-deviation L-inf
    # perturbation per feature, chosen after epsilon=8.0-in-raw-units was
    # found to perturb zero real rows enough to flip a prediction (raw
    # feature scales range from single digits to the thousands, so a fixed
    # raw epsilon is not comparably meaningful across features).
    results = run_full_evaluation(
        X, y, durations_ms, memory_mb, epsilon=1.0, feature_names=FEATURE_COLUMNS, n_clients=3, groups=groups
    )
    print(json.dumps(results, indent=2))

    with open("results_full.json", "w") as f:
        json.dump(results, f, indent=2)
