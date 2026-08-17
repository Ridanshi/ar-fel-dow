import numpy as np
import torch
from torch import nn
from arfel_dow.surrogate import Surrogate


class _Generator(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AdaptiveGANAttacker:
    """Generator trained to produce attack-class feature vectors that the
    surrogate classifies as benign, while staying close to the real
    benign-feature distribution (so the traffic 'reads as legitimate load'
    per the DoW threat model). Methodology ported from GADoT (arXiv:2201.13102)
    and GT-CSAT's WGAN-GP threat generator, simplified to plain adversarial
    loss (no gradient penalty) since our tabular feature space is small and
    CPU-only."""

    def __init__(self, input_dim: int, noise_dim: int, random_state: int = 42):
        torch.manual_seed(random_state)
        self.noise_dim = noise_dim
        self.generator = _Generator(noise_dim, input_dim)
        self._benign_mean = None
        self._benign_std = None
        self._attack_center = None

    def fit(
        self,
        X_benign: np.ndarray,
        X_attack: np.ndarray,
        surrogate: Surrogate,
        epochs: int = 100,
    ) -> "AdaptiveGANAttacker":
        self._benign_mean = X_benign.mean(axis=0)
        self._benign_std = X_benign.std(axis=0) + 1e-6
        self._attack_center = torch.tensor(X_attack.mean(axis=0), dtype=torch.float32)

        optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.01)
        n = len(X_attack)

        for _ in range(epochs):
            optimizer.zero_grad()
            z = torch.randn(n, self.noise_dim)
            generated = self.generator(z) + self._attack_center

            proba = surrogate.forward(generated)
            evasion_loss = proba[:, 1].mean()

            benign_mean_t = torch.tensor(self._benign_mean, dtype=torch.float32)
            benign_std_t = torch.tensor(self._benign_std, dtype=torch.float32)
            plausibility_loss = (((generated - benign_mean_t) / benign_std_t) ** 2).mean()

            loss = evasion_loss + 0.1 * plausibility_loss
            loss.backward()
            optimizer.step()

        return self

    def generate(self, n_samples: int) -> np.ndarray:
        z = torch.randn(n_samples, self.noise_dim)
        with torch.no_grad():
            generated = self.generator(z) + self._attack_center
        return generated.numpy()
