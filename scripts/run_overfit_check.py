"""Direct train-vs-test accuracy gap check.

Sections 6.1/7 argue the 99.42% clean-accuracy figure is not overfitting,
citing held-out group-by-IP evaluation, balanced per-class recall,
federated corroboration, and 5-seed stability — but none of those is a
literal train-accuracy-vs-test-accuracy gap measurement. This script adds
that missing direct check: fit the same StackedEnsembleDetector used
throughout the paper on the train partition, score it on both the train
partition it was fit on and the held-out test partition, and report the
gap, across the same 5 seeds used for the headline multi-seed results.
"""
import json
import time

from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.evaluate import _train_test_split
from arfel_dow.weighting import balanced_sample_weight
from run_experiment import load_and_prepare

SEEDS = [42, 43, 44, 45, 46]

if __name__ == "__main__":
    t0 = time.time()
    print(f"[{time.time()-t0:.0f}s] loading and preparing data...", flush=True)
    X, y, durations_ms, memory_mb, groups, feature_columns = load_and_prepare()
    print(f"[{time.time()-t0:.0f}s] loaded, X shape={X.shape}", flush=True)

    results = []
    for seed in SEEDS:
        print(f"[{time.time()-t0:.0f}s] seed {seed}: splitting...", flush=True)
        X_train, y_train, X_test, y_test, _, _ = _train_test_split(
            X, y, test_size=0.3, random_state=seed, groups=groups
        )
        print(f"[{time.time()-t0:.0f}s] seed {seed}: fitting on {X_train.shape[0]} train rows...", flush=True)
        model = StackedEnsembleDetector(random_state=seed).fit(
            X_train, y_train, sample_weight=balanced_sample_weight(y_train)
        )
        print(f"[{time.time()-t0:.0f}s] seed {seed}: fit done, scoring train...", flush=True)
        train_acc = float((model.predict(X_train) == y_train).mean())
        print(f"[{time.time()-t0:.0f}s] seed {seed}: scoring test...", flush=True)
        test_acc = float((model.predict(X_test) == y_test).mean())
        gap = train_acc - test_acc
        row = {"seed": seed, "train_accuracy": train_acc, "test_accuracy": test_acc, "gap_pp": gap * 100}
        results.append(row)
        print(f"[{time.time()-t0:.0f}s] " + json.dumps(row), flush=True)

    mean_gap = sum(r["gap_pp"] for r in results) / len(results)
    print(f"\nMean train-test gap across {len(SEEDS)} seeds: {mean_gap:.3f} pp")

    with open("results_overfit_check.json", "w") as f:
        json.dump(results, f, indent=2)
