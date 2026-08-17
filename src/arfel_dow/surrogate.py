import numpy as np
import torch
from torch import nn


class Surrogate(nn.Module):
    """Differentiable MLP distilled from the (partly non-differentiable)
    StackedEnsembleDetector, per Papernot et al. (2017) substitute-model
    methodology, so gradient-based attacks (FGSM/PGD/CW) have something to
    take gradients against. Attacks transfer to the real ensemble."""

    def __init__(self, input_dim: int, random_state: int = 42):
        super().__init__()
        torch.manual_seed(random_state)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)

    def fit(self, X: np.ndarray, ensemble_proba: np.ndarray, epochs: int = 200) -> "Surrogate":
        X_t = torch.tensor(X, dtype=torch.float32)
        target_t = torch.tensor(ensemble_proba, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            log_pred = torch.log(self.forward(X_t) + 1e-8)
            loss = loss_fn(log_pred, target_t)
            loss.backward()
            optimizer.step()
        self.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X_t).numpy()
