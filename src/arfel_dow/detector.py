import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class StackedEnsembleDetector:
    """XGBoost + RandomForest + CatBoost + MLP, stacked under a
    logistic-regression meta-learner. The source AR-FEL paper's ensemble is
    XGBoost + RandomForest + GradientBoosting + MLP — this is a stated
    deviation from that architecture, not an exact port: sklearn's
    GradientBoostingClassifier is replaced by CatBoost, a more modern
    gradient-boosting implementation with better built-in regularization,
    keeping the ensemble at four base learners (two boosting variants, one
    bagging, one neural) rather than adding a fifth on top of the
    original four.

    Base-learner hyperparameters are set to standard regularization-aware
    values rather than left at library defaults — this is a light,
    principled tuning pass (bounded tree depth, moderate learning rates,
    L2 regularization on the MLP), not a formal cross-validated
    hyperparameter search. Intended to reduce the gap between training and
    held-out accuracy without claiming a full HPO sweep was run."""

    def __init__(self, random_state: int = 42):
        base_learners = [
            (
                "xgboost",
                XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    max_depth=6,
                    learning_rate=0.05,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    random_state=random_state,
                    max_depth=10,
                    min_samples_leaf=5,
                    n_estimators=200,
                ),
            ),
            (
                "catboost",
                CatBoostClassifier(
                    random_state=random_state,
                    depth=6,
                    learning_rate=0.05,
                    iterations=200,
                    verbose=False,
                ),
            ),
            (
                "mlp",
                MLPClassifier(
                    random_state=random_state,
                    max_iter=1000,
                    hidden_layer_sizes=(64, 32),
                    alpha=0.001,
                ),
            ),
        ]
        self._model = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(random_state=random_state),
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "StackedEnsembleDetector":
        """sample_weight, if given, is forwarded to every base learner that
        supports it (XGBoost, RandomForest, CatBoost, LogisticRegression all
        do; sklearn's StackingClassifier silently skips it for MLPClassifier,
        which doesn't). Lets callers keep the full imbalanced dataset and
        up-weight the minority class instead of downsampling the majority
        class and discarding real data."""
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
