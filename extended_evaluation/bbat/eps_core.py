"""PGD adversarial training at different training epsilons; evaluated against transfer PGD and the black-box search at the
matched epsilon and at epsilon 1, plus clean performance."""
import sys
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from rlib import *


def run_eps(seed, tag, plain_name, Xtr, ytr, Xte, yte, log, eps_list=(0.1, 0.25, 0.5, 1.0), n_eval=1500):
    sc = FeatureScaler().fit(Xtr); Xtr_s = sc.transform(Xtr); lo = Xtr_s.min(0); hi = Xtr_s.max(0)
    plain, _ = cached_fit(plain_name, seed, Xtr, ytr, balanced_sample_weight(ytr), log)
    sur = Surrogate(input_dim=Xtr.shape[1], random_state=seed).fit(Xtr_s, plain.predict_proba(Xtr), epochs=200)
    at_tr = ytr == 1; at = yte == 1; Xat = Xte[at]; Xat_s = sc.transform(Xat)
    sub = np.random.default_rng(seed + 3).choice(len(Xat), size=min(n_eval, len(Xat)), replace=False)
    nf = Xtr.shape[1]; allm = np.ones(nf, dtype=np.float32)
    def attack_rates(model, e):
        Xp = sc.inverse_transform(pgd_attack(sur, Xat_s, np.ones(len(Xat), dtype=np.int64), e, e / 8, 15))
        adv, _ = blackbox_search(model, Xat_s[sub], sc, e, allm, [], lo, hi, 30, np.random.default_rng(seed + 5))
        return float((model.predict(Xp) == 1).mean()), float((model.predict(sc.inverse_transform(adv)) == 1).mean())
    def clean(model):
        p = model.predict(Xte); return {"acc": float((p == yte).mean()), "benign_recall": float((p[~at] == 0).mean()), "attack_recall": float((p[at] == 1).mean())}
    out = {"plain": {"clean": clean(plain)}}
    for e in (0.25, 0.5, 1.0):
        t, b = attack_rates(plain, e); out["plain"][f"transfer_e{e}"] = t; out["plain"][f"bb_e{e}"] = b
    for et in eps_list:
        Xadv = sc.inverse_transform(pgd_attack(sur, Xtr_s[at_tr], np.ones(at_tr.sum(), dtype=np.int64), et, et / 8, 15))
        Xa = np.vstack([Xtr, Xadv]); ya = np.concatenate([ytr, np.ones(len(Xadv), dtype=np.int64)])
        m, _ = cached_fit(f"{tag}_pgdeps{et}", seed, Xa, ya, balanced_sample_weight(ya), log)
        r = {"clean": clean(m)}
        for e in sorted(set((0.25, 0.5, 1.0, et))):
            t, b = attack_rates(m, e); r[f"transfer_e{e}"] = t; r[f"bb_e{e}"] = b
        out[f"train_eps_{et}"] = r
        log(f"{tag} train_eps {et}: acc {r['clean']['acc']:.3f} atkrec {r['clean']['attack_recall']:.3f} transfer_e1 {r['transfer_e1.0']:.3f} bb_e1 {r['bb_e1.0']:.3f}")
    return out
