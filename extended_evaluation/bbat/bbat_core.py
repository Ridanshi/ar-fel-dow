"""Iterated black-box adversarial training: attack the current detector with the query-only search on training
attack rows, add the evasive rows as attacks, refit, repeat. After each round the attacker re-attacks the new model
on held-out attack rows (adaptive evaluation)."""
import sys
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from rlib import *


def _bb(model, sc, Xs, lo, hi, iters, seed):
    nf = Xs.shape[1]
    adv, _ = blackbox_search(model, Xs, sc, EPS, np.ones(nf, dtype=np.float32), [], lo, hi, iters, np.random.default_rng(seed))
    return adv


def metrics(model, sc, Xte, yte, lo, hi, seed, n_eval=1500):
    at = yte == 1
    p = model.predict(Xte)
    r = {"acc": float((p == yte).mean()), "benign_recall": float((p[~at] == 0).mean()), "attack_recall": float((p[at] == 1).mean())}
    Xat = Xte[at]
    sub = np.random.default_rng(seed + 3).choice(len(Xat), size=min(n_eval, len(Xat)), replace=False)
    Xs = sc.transform(Xat[sub])
    for it in (30, 60):
        adv = _bb(model, sc, Xs, lo, hi, it, seed + 5)
        r[f"bb{it}"] = float((model.predict(sc.inverse_transform(adv)) == 1).mean())
    return r


def run_bbat(seed, tag, plain_name, Xtr, ytr, Xte, yte, log, rounds=3, n_adv=3000):
    sc = FeatureScaler().fit(Xtr); Xtr_s = sc.transform(Xtr); lo = Xtr_s.min(0); hi = Xtr_s.max(0)
    plain, _ = cached_fit(plain_name, seed, Xtr, ytr, balanced_sample_weight(ytr), log)
    out = {"round0": metrics(plain, sc, Xte, yte, lo, hi, seed)}
    log(f"{tag} round0 " + " ".join(f"{k}={v:.3f}" for k, v in out["round0"].items()))
    model = plain; adv_all = []
    at_idx = np.flatnonzero(ytr == 1)
    rng = np.random.default_rng(seed + 17)
    for r in range(1, rounds + 1):
        pick = rng.choice(at_idx, size=min(n_adv, len(at_idx)), replace=False)
        adv = sc.inverse_transform(_bb(model, sc, Xtr_s[pick], lo, hi, 30, seed + 100 + r))
        adv_all.append(adv)
        Xa = np.vstack([Xtr] + adv_all); ya = np.concatenate([ytr] + [np.ones(len(a), dtype=np.int64) for a in adv_all])
        model, _ = cached_fit(f"{tag}_bbat_r{r}", seed, Xa, ya, balanced_sample_weight(ya), log)
        out[f"round{r}"] = metrics(model, sc, Xte, yte, lo, hi, seed)
        log(f"{tag} round{r} " + " ".join(f"{k}={v:.3f}" for k, v in out[f"round{r}"].items()))
    return out
