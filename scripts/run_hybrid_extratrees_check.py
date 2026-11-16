"""Exploratory check requested during review: does adding a 5th base
learner, this time an Extremely Randomized Trees (ExtraTrees) learner,
to the existing 4-learner stacked ensemble (XGBoost + RandomForest +
CatBoost + MLP) improve on the current pipeline? Unlike the two prior
hybrid checks (attention-based, GRU-based — both null), this tests
additional tree-ensemble diversity (a third, differently-randomized
bagging variant) rather than a sequence/attention neural architecture.

Per instructions: if the hybrid variant measurably beats the current
ensemble, it goes in the paper. If not, this script and its result stay
out of the paper entirely.

Scoped to a 50k-row sample, single seed (42) — an exploratory check, not
a repeat of the 5-seed headline run. Uses run_evaluation (clean + PGD +
GAN metrics) rather than run_full_evaluation, skipping the federated
simulation, which is orthogonal to the base-learner question being asked
here and would roughly double runtime for no additional signal.
"""
import json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import arfel_dow.evaluate as evaluate_module
import arfel_dow.adversarial_training as adv_training_module
from run_experiment import load_and_prepare
from arfel_dow.evaluate import run_evaluation

SEED = 42
SAMPLE_SIZE = 50_000


class HybridEnsembleDetector:
    """Same 4 base learners as StackedEnsembleDetector, plus a 5th:
    an ExtraTreesClassifier (extremely randomized trees)."""

    def __init__(self, random_state: int = 42):
        base_learners = [
            ("xgboost", XGBClassifier(
                random_state=random_state, eval_metric="logloss", max_depth=6,
                learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            )),
            ("random_forest", RandomForestClassifier(
                random_state=random_state, max_depth=10, min_samples_leaf=5, n_estimators=200,
            )),
            ("catboost", CatBoostClassifier(
                random_state=random_state, depth=6, learning_rate=0.05, iterations=200, verbose=False,
            )),
            ("mlp", MLPClassifier(
                random_state=random_state, max_iter=1000, hidden_layer_sizes=(64, 32), alpha=0.001,
            )),
            ("extra_trees", ExtraTreesClassifier(
                random_state=random_state, max_depth=10, min_samples_leaf=5, n_estimators=200,
            )),
        ]
        self._model = StackingClassifier(
            estimators=base_learners, final_estimator=LogisticRegression(random_state=random_state)
        )

    def fit(self, X, y, sample_weight=None):
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


if __name__ == "__main__":
    X, y, durations_ms, memory_mb, groups, feature_columns = load_and_prepare(sample_size=SAMPLE_SIZE)

    print("=== Baseline: 4-learner ensemble (current pipeline) ===", flush=True)
    results_baseline = run_evaluation(
        X, y, durations_ms, memory_mb, epsilon=1.0, random_state=SEED, groups=groups,
    )
    print(json.dumps(results_baseline, indent=2), flush=True)

    print("=== Hybrid: 5-learner ensemble (+ ExtraTrees) ===", flush=True)
    evaluate_module.StackedEnsembleDetector = HybridEnsembleDetector
    adv_training_module.StackedEnsembleDetector = HybridEnsembleDetector
    results_hybrid = run_evaluation(
        X, y, durations_ms, memory_mb, epsilon=1.0, random_state=SEED, groups=groups,
    )
    print(json.dumps(results_hybrid, indent=2), flush=True)

    with open("results_hybrid_extratrees_check.json", "w") as f:
        json.dump({"baseline": results_baseline, "hybrid": results_hybrid}, f, indent=2)

    print("\n=== DELTA (hybrid - baseline) ===", flush=True)
    for key in results_baseline:
        delta = results_hybrid[key] - results_baseline[key]
        print(f"{key}: {delta:+.4f}", flush=True)
