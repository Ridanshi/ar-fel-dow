import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin


class _AttentionNet(nn.Module):
    """Treats the tabular feature vector as a sequence of scalar tokens (one
    per feature), embeds each into d_model dimensions, applies multi-head
    self-attention across features, then classifies from the pooled
    representation. Inspired by MHARNN-DoWAD's multi-head-attention approach
    to DoW detection, adapted here to a static feature vector rather than a
    recurrent/sequential input, since DoWTS rows are independent invocations
    rather than time-windowed sequences."""

    def __init__(self, n_features: int, d_model: int = 16, n_heads: int = 2, random_state: int = 42):
        super().__init__()
        torch.manual_seed(random_state)
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x.unsqueeze(-1)) + self.pos
        attended, _ = self.attn(tokens, tokens, tokens)
        pooled = attended.mean(dim=1)
        return self.head(pooled)


class AttentionEnsembleMember(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around _AttentionNet so it can slot into
    StackingClassifier as an additional base learner alongside
    XGBoost/RandomForest/CatBoost/MLP. Accepts sample_weight and applies it
    as a per-sample loss weight (this sklearn version raises rather than
    silently skipping sample_weight for estimators whose fit() doesn't
    declare it, unlike the older behavior MLPClassifier's integration
    relied on). Inherits BaseEstimator/ClassifierMixin for sklearn's tag
    system (required by StackingClassifier's internal is_classifier()
    check) and get_params/set_params/clone support."""

    def __init__(self, random_state: int = 42, epochs: int = 60):
        self.random_state = random_state
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "AttentionEnsembleMember":
        self.classes_ = np.unique(y)
        self._net = _AttentionNet(n_features=X.shape[1], random_state=self.random_state)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        w_t = torch.tensor(sample_weight, dtype=torch.float32)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=0.005)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        self._net.train()
        batch_size = 4096
        n = len(X)
        for _ in range(self.epochs):
            perm = np.random.default_rng(self.random_state).permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                logits = self._net(X_t[idx])
                per_sample_loss = loss_fn(logits, y_t[idx])
                loss = (per_sample_loss * w_t[idx]).mean()
                loss.backward()
                optimizer.step()
        self._net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(self._net(X_t), dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
