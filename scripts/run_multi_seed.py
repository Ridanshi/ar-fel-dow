"""Multi-seed statistical rigor for the paper's headline results.

Every number previously reported in the paper came from a single run at
random_state=42 — no repeated trials, no variance estimate. This script
reruns the full pipeline (data loading is seed-independent and done once;
the group-by-IP split, ensemble fits, and attacks all depend on
random_state and are rerun per seed) across N_SEEDS seeds and reports
mean +/- standard deviation for every metric run_full_evaluation returns.

CPU-only, no GPU. Each seed reruns the same ~6 ensemble fits as a single
run_experiment.py invocation, so total runtime is roughly N_SEEDS times a
single full run — expect this to take a while; run in the background.
"""
import json
import numpy as np
from run_experiment import load_and_prepare
from arfel_dow.evaluate import run_full_evaluation

SEEDS = [42, 43, 44, 45, 46]

if __name__ == "__main__":
    # Feature engineering and data loading don't depend on random_state,
    # so this is done once and reused across all seeds rather than
    # reloading/re-parsing the 187k-row CSV five times.
    X, y, durations_ms, memory_mb, groups, feature_columns = load_and_prepare()

    all_results = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        results = run_full_evaluation(
            X, y, durations_ms, memory_mb,
            epsilon=1.0, feature_names=feature_columns, n_clients=3,
            groups=groups, random_state=seed,
        )
        results["seed"] = seed
        print(json.dumps(results, indent=2), flush=True)
        all_results.append(results)

    with open("results_multi_seed.json", "w") as f:
        json.dump(all_results, f, indent=2)

    numeric_keys = [
        k for k in all_results[0]
        if k != "seed" and k != "sample_narrative" and isinstance(all_results[0][k], (int, float))
    ]
    summary = {}
    for key in numeric_keys:
        values = np.array([r[key] for r in all_results], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std()), "values": values.tolist()}

    print("\n=== SUMMARY (mean +/- std across seeds) ===")
    for key, stat in summary.items():
        print(f"{key}: {stat['mean']:.4f} +/- {stat['std']:.4f}")

    with open("results_multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
