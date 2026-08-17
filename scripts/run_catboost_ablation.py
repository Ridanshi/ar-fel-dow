"""Isolated ablation: CatBoost vs. GradientBoostingClassifier as the third
base learner, holding everything else in the pipeline fixed.

Section 4.3 of the paper states the CatBoost-for-GradientBoostingClassifier
substitution was evaluated alongside several concurrent methodology changes
(class-weighting, feature expansion, the group-by-IP split fix) and its own
contribution was never separately isolated. This script isolates it: same
group-by-IP split, same data, same hyperparameters on every other base
learner, same seed — the only difference between the two runs below is
which gradient-boosting implementation sits in the third ensemble slot.

Uses monkeypatching rather than duplicating run_full_evaluation's pipeline
logic: StackedEnsembleDetector is imported by name into evaluate.py,
adversarial_training.py, and federated.py at their own module scope, so
rebinding those names to an alternate class is picked up at call time
(Python resolves module-global names when the function runs, not when it
was defined) without touching the tested source files.
"""
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import arfel_dow.evaluate as evaluate_module
import arfel_dow.adversarial_training as adv_training_module
import arfel_dow.federated as federated_module
from arfel_dow.detector import StackedEnsembleDetector as CatBoostEnsembleDetector
from run_experiment import load_and_prepare
from arfel_dow.evaluate import run_full_evaluation

SEED = 42


class GBMEnsembleDetector:
    """Same four-learner stacking architecture as StackedEnsembleDetector,
    with scikit-learn's GradientBoostingClassifier in place of CatBoost —
    the AR-FEL source paper's original third base learner (Section 2.3),
    reconstructed here only for this isolated ablation."""

    def __init__(self, random_state: int = 42):
        base_learners = [
            ("xgboost", XGBClassifier(
                random_state=random_state, eval_metric="logloss", max_depth=6,
                learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            )),
            ("random_forest", RandomForestClassifier(
                random_state=random_state, max_depth=10, min_samples_leaf=5, n_estimators=200,
            )),
            ("gradient_boosting", GradientBoostingClassifier(
                random_state=random_state, max_depth=3, learning_rate=0.05, n_estimators=200,
            )),
            ("mlp", MLPClassifier(
                random_state=random_state, max_iter=1000, hidden_layer_sizes=(64, 32), alpha=0.001,
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
    X, y, durations_ms, memory_mb, groups, feature_columns = load_and_prepare()

    print("=== CatBoost (current pipeline) ===", flush=True)
    results_catboost = run_full_evaluation(
        X, y, durations_ms, memory_mb, epsilon=1.0, feature_names=feature_columns,
        n_clients=3, groups=groups, random_state=SEED,
    )
    print(json.dumps(results_catboost, indent=2), flush=True)

    print("=== GradientBoostingClassifier (isolation) ===", flush=True)
    evaluate_module.StackedEnsembleDetector = GBMEnsembleDetector
    adv_training_module.StackedEnsembleDetector = GBMEnsembleDetector
    federated_module.StackedEnsembleDetector = GBMEnsembleDetector
    results_gbm = run_full_evaluation(
        X, y, durations_ms, memory_mb, epsilon=1.0, feature_names=feature_columns,
        n_clients=3, groups=groups, random_state=SEED,
    )
    print(json.dumps(results_gbm, indent=2), flush=True)

    with open("results_catboost_ablation.json", "w") as f:
        json.dump({"catboost": results_catboost, "gradient_boosting": results_gbm}, f, indent=2)

    print("\n=== DELTA (CatBoost - GradientBoosting) ===")
    for key in results_catboost:
        if key == "sample_narrative":
            continue
        delta = results_catboost[key] - results_gbm[key]
        print(f"{key}: {delta:+.4f}")
