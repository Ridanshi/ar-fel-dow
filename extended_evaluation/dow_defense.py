"""Billing-aware mitigation: per-identity evidence accumulation with sticky blocking.
Evaluated causally per source IP on the held-out DoWTS test split, with the cached
undefended and PGD-trained detectors. Also measures dollars leaked before blocking."""
import sys, json, pickle
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from rlib import *
from run_experiment import load_and_prepare
from arfel_dow.evaluate import _train_test_split
seed = int(sys.argv[1])
X, y, dur, mem, groups, cols = load_and_prepare()
Xtr, ytr, Xte, yte, tr, te = _train_test_split(X, y, 0.3, seed, groups)
CNT = cols.index("ip_request_sequence_number")
g = groups[te]
cost = invocation_costs(dur[te], mem[te])
cost_cap = invocation_costs(dur[te], np.minimum(mem[te], 10240.0))
order = np.lexsort((Xte[:, CNT], g))
Xo, yo, go, co, cco = Xte[order], yte[order], g[order], cost[order], cost_cap[order]
starts = np.flatnonzero(np.r_[True, go[1:] != go[:-1]])
ends = np.r_[starts[1:], len(go)]
LOG = lambda m: print(time.strftime("%H:%M:%S"), f"[s{seed}]", m, flush=True)


def load(name):
    return pickle.load(open(f"{CACHE}/s{seed}_{name}.pkl", "rb"))["model"]


def decide(p, lab, c, cc, idx_ranges, w, tau):
    """p: per-request attack probability in identity order. Returns metrics for
    per-request thresholding (w=0) or accumulation over the last w requests."""
    blocked = np.zeros(len(p), bool)
    for a, b in idx_ranges:
        seg = p[a:b]
        if w == 0:
            blocked[a:b] = seg > 0.5
        else:
            cs = np.r_[0.0, np.cumsum(seg)]
            k = np.arange(1, len(seg) + 1)
            lo = np.maximum(0, k - w)
            s = (cs[k] - cs[lo]) / (k - lo)
            hit = np.flatnonzero(s > tau)
            if len(hit):
                blocked[a + hit[0]:b] = True
    at = lab == 1
    return {"attack_blocked": float(blocked[at].mean()), "usd_leaked": float(c[at & ~blocked].sum()),
            "usd_leaked_capped": float(cc[at & ~blocked].sum()), "usd_attack_total": float(c[at].sum()),
            "benign_blocked": float(blocked[~at].mean()),
            "benign_ids_blocked": float(np.mean([blocked[a:b][lab[a:b] == 0].any() for a, b in idx_ranges if (lab[a:b] == 0).any()]))}


R = {"seed": seed}
rng = np.random.default_rng(seed)
for mname in ("plain", "pgd_at"):
    mod = load(mname)
    ranges = list(zip(starts, ends))
    p = mod.predict_proba(Xo)[:, 1]
    R[mname] = {"clean": {f"w{w}": decide(p, yo, co, cco, ranges, w, 0.5) for w in (0, 5, 10, 20)}}
    for r in (3, 10, 30, 100):
        Xr = Xo.copy()
        sel = []
        nr = []
        for a, b in ranges:
            if yo[a] == 1:
                pos = np.arange(b - a)
                Xr[a:b, CNT] = pos % r
                nr += [(a + i, min(a + i + r, b)) for i in range(0, b - a, r)]
            else:
                nr.append((a, b))
        nr.sort()
        pr = mod.predict_proba(Xr)[:, 1]
        R[mname][f"rot{r}"] = {f"w{w}": decide(pr, yo, co, cco, nr, w, 0.5) for w in (0, 5, 10, 20)}
    LOG(f"{mname}: clean w0 leaked ${R[mname]['clean']['w0']['usd_leaked']:.2f} w10 ${R[mname]['clean']['w10']['usd_leaked']:.2f} | rot10 w0 ${R[mname]['rot10']['w0']['usd_leaked']:.2f} w10 ${R[mname]['rot10']['w10']['usd_leaked']:.2f}")
json.dump(R, open(f"D:/Cloud RP/extended_evaluation/results/seed{seed}_defense.json", "w"), indent=1)
LOG("DEFENSE DONE")
