import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin


class _GRUNet(nn.Module):
    """Treats the tabular feature vector as a length-n_features sequence of
    scalar timesteps and runs it through a GRU, taking the final hidden
    state as the pooled representation before classifying. Inspired by
    CDMDoW-AMOAFL's GRU/TCN/convolutional-autoencoder ensemble, adapted to
    a static feature vector rather than genuine time-series telemetry,
    since DoWTS rows are independent invocations, not per-tenant sequences."""

    def __init__(self, n_features: int, hidden_size: int = 16, random_state: int = 42):
        super().__init__()
        torch.manual_seed(random_state)
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)
        _, h_n = self.gru(seq)
        return self.head(h_n[-1])


class GRUEnsembleMember(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around _GRUNet, same interface/training
    loop as AttentionEnsembleMember, for slotting into StackingClassifier
    as a base learner inspired by CDMDoW-AMOAFL [2]."""

    def __init__(self, random_state: int = 42, epochs: int = 60):
        self.random_state = random_state
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "GRUEnsembleMember":
        self.classes_ = np.unique(y)
        self._net = _GRUNet(n_features=X.shape[1], random_state=self.random_state)
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
