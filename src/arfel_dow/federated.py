import numpy as np
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.weighting import balanced_sample_weight


class FederatedEnsembleTrainer:
    """Trains one full StackedEnsembleDetector per simulated tenant, each on
    that tenant's own data only — no raw data crosses tenant boundaries, only
    the fitted model is used afterward. Predictions are aggregated by
    averaging predict_proba across tenants ('federated ensemble
    aggregation'), rather than classic FedAvg parameter-averaging, since
    tree-based base learners (XGBoost/RandomForest/GradientBoosting) have no
    parameter vector to average the way a neural net does.

    Anti-poisoning property: a single malicious tenant's poisoned labels
    only ever influence one of the N voting models, rather than being mixed
    directly into a single pooled training set — diluting its influence to
    roughly 1/N of the aggregate vote."""

    def __init__(self):
        self._client_models: list[StackedEnsembleDetector] = []

    def fit(
        self,
        client_partitions: list[tuple[np.ndarray, np.ndarray]],
        random_state: int = 42,
    ) -> "FederatedEnsembleTrainer":
        self._client_models = []
        for i, (X_client, y_client) in enumerate(client_partitions):
            model = StackedEnsembleDetector(random_state=random_state + i)
            model.fit(X_client, y_client, sample_weight=balanced_sample_weight(y_client))
            self._client_models.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        client_probas = np.stack([model.predict_proba(X) for model in self._client_models])
        return client_probas.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
