import numpy as np
import torch
from arfel_dow.surrogate import Surrogate


def fgsm_attack(surrogate: Surrogate, X: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long)
    proba = surrogate.forward(X_t)
    loss = torch.nn.functional.nll_loss(torch.log(proba + 1e-8), y_t)
    loss.backward()
    perturbation = torch.clamp(epsilon * X_t.grad.sign(), min=-epsilon, max=epsilon)
    X_adv = (X_t + perturbation).detach().numpy()
    return X_adv


def pgd_attack(
    surrogate: Surrogate,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
    steps: int,
) -> np.ndarray:
    X_orig = torch.tensor(X, dtype=torch.float32)
    X_adv = X_orig.clone().detach()
    y_t = torch.tensor(y, dtype=torch.long)

    for _ in range(steps):
        X_adv.requires_grad_(True)
        proba = surrogate.forward(X_adv)
        loss = torch.nn.functional.nll_loss(torch.log(proba + 1e-8), y_t)
        grad = torch.autograd.grad(loss, X_adv)[0]
        X_adv = X_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
        X_adv = (X_orig + delta).detach()

    return X_adv.numpy()


def cw_attack(
    surrogate: Surrogate,
    X: np.ndarray,
    y: np.ndarray,
    c: float,
    steps: int,
) -> np.ndarray:
    """Simplified Carlini & Wagner L2 attack: minimize
    ||delta||^2 + c * (confidence of the true class), via gradient descent
    on an unconstrained delta (no box constraint, since features here are
    not bounded to [0, 1] like pixels)."""
    X_orig = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    delta = torch.zeros_like(X_orig, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.1)

    for _ in range(steps):
        optimizer.zero_grad()
        X_adv = X_orig + delta
        proba = surrogate.forward(X_adv)
        true_class_proba = proba.gather(1, y_t.view(-1, 1)).squeeze(1)
        loss = (delta ** 2).sum(dim=1).mean() + c * true_class_proba.mean()
        loss.backward()
        optimizer.step()

    return (X_orig + delta).detach().numpy()
