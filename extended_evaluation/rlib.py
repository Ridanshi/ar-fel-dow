"""Extended-evaluation experiments: shared helpers (attacks, defenses, evaluation)."""
import sys, os, time, pickle, io
sys.path.insert(0, "D:/Cloud RP/scripts")
sys.path.insert(0, "D:/Cloud RP/src")
os.chdir("D:/Cloud RP")
import numpy as np
import psutil
import torch
from torch import nn
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.surrogate import Surrogate
from arfel_dow.scaling import FeatureScaler
from arfel_dow.attacks_gradient import pgd_attack
from arfel_dow.attacks_gan import AdaptiveGANAttacker
from arfel_dow.weighting import balanced_sample_weight
from arfel_dow.pricing import AWSLambdaPricing
from arfel_dow.evaluate import _train_test_split

CACHE = "D:/Cloud RP/extended_evaluation/cache"
EPS = 1.0


def rss_mb():
    return psutil.Process().memory_info().rss / 2**20


def timed_fit(X, y, w, seed):
    t = time.time()
    m = StackedEnsembleDetector(random_state=seed).fit(X, y, sample_weight=w)
    return m, time.time() - t


def cached_fit(name, seed, X, y, w, log):
    path = f"{CACHE}/s{seed}_{name}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log(f"[cache] loaded {name} (fit {obj['fit_seconds']:.0f}s)")
        return obj["model"], obj["fit_seconds"]
    m, sec = timed_fit(X, y, w, seed)
    with open(path, "wb") as f:
        pickle.dump({"model": m, "fit_seconds": sec}, f)
    log(f"[fit] {name}: {sec:.0f}s (rss {rss_mb():.0f} MB)")
    return m, sec


# ---------------------------------------------------------------- features
def feature_groups(cols):
    idx = {c: i for i, c in enumerate(cols)}
    direct = [idx["SubmitTime"]]
    onedir = [idx["ip_request_sequence_number"]]
    coarse = [idx[c] for c in ("FunctionDuration", "ActiveFunctionsAtRequest", "ActiveFunctionsAtResponse", "RTT")]
    return idx, direct, onedir, coarse


def make_masks(cols):
    idx, direct, onedir, coarse = feature_groups(cols)
    n = len(cols)
    def m(ix):
        v = np.zeros(n, dtype=np.float32); v[ix] = 1.0; return v
    return {
        "all": np.ones(n, dtype=np.float32),
        "influenceable": m(direct + onedir + coarse),
        "direct": m(direct + onedir),
    }, onedir


# ---------------------------------------------------------------- attacks
def masked_pgd(surrogate, Xs, y, eps, alpha, steps, mask, increase_only, lo, hi):
    """PGD in standardized space restricted to `mask` features; features in
    `increase_only` may only increase; result clipped to [lo, hi] (training range)."""
    X0 = torch.tensor(Xs, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    lo_t = torch.tensor(lo, dtype=torch.float32); hi_t = torch.tensor(hi, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    inc = torch.zeros_like(mask_t)
    for i in increase_only:
        inc[i] = 1.0
    Xa = X0.clone()
    for _ in range(steps):
        Xa.requires_grad_(True)
        p = surrogate.forward(Xa)
        loss = torch.nn.functional.nll_loss(torch.log(p + 1e-8), yt)
        g = torch.autograd.grad(loss, Xa)[0]
        Xa = Xa.detach() + alpha * g.sign() * mask_t
        d = torch.clamp(Xa - X0, -eps, eps)
        d = torch.where(inc > 0, torch.clamp(d, min=0.0), d)
        Xa = torch.min(torch.max(X0 + d * mask_t, lo_t), hi_t).detach()
    return Xa.numpy()


def blackbox_search(model, Xs, scaler, eps, mask, increase_only, lo, hi, iters, rng):
    """Score-based random-search attack directly on the deployed model (queries
    only: no gradients, no surrogate). Each iteration proposes, per row, a move of a
    random third of the allowed features to a corner of the eps-box and keeps the
    proposal when it lowers P(attack). Returns adversarial rows (standardized) and
    the number of queries per row."""
    n, d = Xs.shape
    allowed = np.where(mask > 0)[0]
    inc = set(increase_only)

    def proba(Z):
        return model.predict_proba(scaler.inverse_transform(Z))[:, 1]

    cur = Xs.copy()
    p_cur = proba(cur)
    queries = 1
    for _ in range(iters):
        sign = rng.choice([-1.0, 1.0], size=(n, len(allowed)))
        sel = rng.random((n, len(allowed))) < 0.33
        delta = cur - Xs
        for j, f in enumerate(allowed):
            s_j = np.abs(sign[:, j]) if f in inc else sign[:, j]
            delta[:, f] = np.where(sel[:, j], s_j * eps, delta[:, f])
        prop = np.clip(Xs + delta, lo, hi)
        p_prop = proba(prop)
        queries += 1
        better = p_prop < p_cur
        cur[better] = prop[better]
        p_cur[better] = p_prop[better]
    return cur, queries


class IndepSurrogate(nn.Module):
    """Independently constructed substitute: different architecture, different
    seed, trained on hard labels queried from the target model."""
    def __init__(self, d, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)
    def fit_hard(self, Xs, yhard, epochs=300):
        X_t = torch.tensor(Xs, dtype=torch.float32); y_t = torch.tensor(yhard, dtype=torch.long)
        opt = torch.optim.Adam(self.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.net(X_t), y_t)
            loss.backward(); opt.step()
        self.eval(); return self


class SmoothedModel:
    """Gaussian-noise smoothing wrapper (randomized-smoothing style prediction)."""
    def __init__(self, base, scaler, sigma, m=8, seed=0):
        self.base, self.scaler, self.sigma, self.m = base, scaler, sigma, m
        self.rng = np.random.default_rng(seed)
    def predict_proba(self, X):
        Z = self.scaler.transform(X)
        acc = 0
        for _ in range(self.m):
            acc = acc + self.base.predict_proba(self.scaler.inverse_transform(Z + self.rng.normal(0, self.sigma, Z.shape)))
        return acc / self.m
    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def acc_attack(model, X):
    return float((model.predict(X) == 1).mean())


def distill(model, scaler, Xtr, seed, n=40000, epochs=200):
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(Xtr), size=min(n, len(Xtr)), replace=False)
    Xs = scaler.transform(Xtr[ix])
    sur = Surrogate(input_dim=Xtr.shape[1], random_state=seed + 11)
    sur.fit(Xs, model.predict_proba(Xtr[ix]), epochs=epochs)
    return sur


def invocation_costs(dur_ms, mem_mb):
    p = AWSLambdaPricing()
    return (dur_ms / 1000.0) * (mem_mb / 1024.0) * p.COMPUTE_COST_PER_GB_SECOND + p.REQUEST_COST_PER_INVOCATION


def frg(y_true_attack_rows_pred_clean, pred_adv, costs):
    missed_clean = y_true_attack_rows_pred_clean == 0
    missed_adv = pred_adv == 0
    return float(costs[missed_adv].sum() - costs[missed_clean].sum())
