"""Before/after effect of the two evaluation-methodology corrections.

(a) Split leakage: the identical detector is trained and scored under a plain
    row-level shuffle instead of the group-by-IP split, with and without the
    engineered per-IP request count.
(b) Attack-row reuse: the robust ensemble is augmented with PGD copies of the
    very attack rows that are then attacked for evaluation, instead of a
    disjoint train subset (the correct protocol).

One JSON per seed is written to results_pitfall_ablation_seed<N>.json, so an
interrupted run resumes at the next missing seed.
"""
import json
import os
import sys
import numpy as np
from run_experiment import load_and_prepare
from arfel_dow.evaluate import _train_test_split
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.weighting import balanced_sample_weight
from arfel_dow.surrogate import Surrogate
from arfel_dow.scaling import FeatureScaler
from arfel_dow.attacks_gradient import pgd_attack
from arfel_dow.attacks_gan import AdaptiveGANAttacker
from arfel_dow.adversarial_training import train_robust_ensemble

SEEDS = [int(s) for s in sys.argv[1:]] or [42, 43, 44, 45, 46]
EPS = 1.0


def split_scores(X, y, groups, seed):
    Xtr, ytr, Xte, yte, _, _ = _train_test_split(X, y, 0.3, seed, groups=groups)
    m = StackedEnsembleDetector(random_state=seed).fit(Xtr, ytr, sample_weight=balanced_sample_weight(ytr))
    p = m.predict(Xte)
    return {"acc": float((p == yte).mean()),
            "attack_recall": float((p[yte == 1] == 1).mean()),
            "benign_recall": float((p[yte == 0] == 0).mean())}


if __name__ == "__main__":
    X, y, dur, mem, groups, cols = load_and_prepare()
    Xn, yn, _, _, groups_n, cols_n = load_and_prepare(include_ip_frequency=False)
    for seed in SEEDS:
        out = f"results_pitfall_ablation_seed{seed}.json"
        if os.path.exists(out):
            continue
        print(f"=== seed {seed} ===", flush=True)
        res = {"seed": seed}
        # (a) split leakage
        res["group_with_count"] = None  # headline figure comes from results_multi_seed.json
        res["row_with_count"] = split_scores(X, y, None, seed)
        print("row_with_count", res["row_with_count"], flush=True)
        res["group_no_count"] = split_scores(Xn, yn, groups_n, seed)
        print("group_no_count", res["group_no_count"], flush=True)
        res["row_no_count"] = split_scores(Xn, yn, None, seed)
        print("row_no_count", res["row_no_count"], flush=True)
        # (b) attack-row reuse, on the correct group split
        Xtr, ytr, Xte, yte, _, _ = _train_test_split(X, y, 0.3, seed, groups=groups)
        plain = StackedEnsembleDetector(random_state=seed).fit(Xtr, ytr, sample_weight=balanced_sample_weight(ytr))
        scaler = FeatureScaler().fit(Xtr)
        sur = Surrogate(input_dim=X.shape[1], random_state=seed)
        sur.fit(scaler.transform(Xtr), plain.predict_proba(Xtr), epochs=200)
        Xa_te = Xte[yte == 1]
        ya_te = yte[yte == 1]
        adv = scaler.inverse_transform(pgd_attack(sur, scaler.transform(Xa_te), ya_te, epsilon=EPS, alpha=EPS / 8, steps=15))
        gan = AdaptiveGANAttacker(input_dim=X.shape[1], noise_dim=8, random_state=seed)
        gan.fit(X_benign=scaler.transform(Xtr[ytr == 0]), X_attack=scaler.transform(Xtr[ytr == 1]), surrogate=sur, epochs=100)
        Xg = scaler.inverse_transform(gan.generate(n_samples=len(Xa_te)))
        res["plain_pgd"] = float((plain.predict(adv) == ya_te).mean())
        res["plain_gan"] = float((plain.predict(Xg) == 1).mean())
        # reuse: augmentation rows ARE the evaluated attack rows
        reuse = train_robust_ensemble(Xtr, ytr, Xa_te, ya_te, sur, scaler, epsilon=EPS, random_state=seed)
        res["reuse_pgd"] = float((reuse.predict(adv) == ya_te).mean())
        res["reuse_gan"] = float((reuse.predict(Xg) == 1).mean())
        res["reuse_clean"] = float((reuse.predict(Xte) == yte).mean())
        print({k: v for k, v in res.items() if k.startswith(("plain", "reuse"))}, flush=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
