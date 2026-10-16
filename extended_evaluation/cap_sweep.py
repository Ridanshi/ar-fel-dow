"""Identity-independent spend cap vs attack intensity on a synthetic workload (bursty, heavy-tailed client volume).
Calibrate on benign spend in the first half; evaluate the second half. No model needed."""
import sys, json
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
import numpy as np
from synth_benign import benign
W = 300.0
SPAN = 17000.0
NB = 20000


def price(dur_ms, mem=512.0):
    return (dur_ms / 1000.0) * (mem / 1024.0) * 0.0000166667 + 0.20 / 1e6


def served_sim(t, c, admit, B):
    q = []; head = 0; spend = 0.0; out = np.zeros(len(t), bool)
    for i in range(len(t)):
        while head < len(q) and t[q[head]] <= t[i] - W:
            spend -= c[q[head]]; head += 1
        if admit[i] and spend < B:
            out[i] = True; q.append(i); spend += c[i]
    return out


res = []
for seed in (42, 43, 44, 45, 46):
    for ratio in (0.0, 2.34, 11.7, 46.8):
        rng = np.random.default_rng(seed)
        st, dur, _ = benign(rng, NB, SPAN, dur_median_ms=300.0, dur_sigma=1.6, cap_ms=60000.0)
        tp = [st]; dp = [dur]; yp = [np.zeros(len(st), int)]
        nbots = 100
        na = int(NB * ratio); per = max(na // nbots, 1) if ratio > 0 else 0
        for j in range(nbots if ratio > 0 else 0):
            gap = rng.lognormal(np.log(max(SPAN / (per * 3.0), 0.5)), 0.5)
            s0 = rng.uniform(0, max(SPAN - gap * per, 1.0))
            ts = s0 + np.cumsum(rng.exponential(gap, per)); ts = ts[ts < SPAN]
            tp.append(ts); dp.append(dur[rng.integers(0, NB, len(ts))]); yp.append(np.ones(len(ts), int))
        t = np.concatenate(tp); d = np.concatenate(dp); y = np.concatenate(yp)
        o = np.argsort(t, kind="stable"); t = t[o]; d = d[o]; y = y[o]
        c = price(d); half = SPAN / 2
        sv = served_sim(t, c, (t < half) & (y == 0), 1e9)
        idx = np.flatnonzero(sv); qq = []; rs = 0.0; sp = []
        for i in idx:
            qq.append(i); rs += c[i]
            while t[qq[0]] <= t[i] - W: rs -= c[qq.pop(0)]
            sp.append(rs)
        base = float(max(sp))
        ev = t >= half; at = (y == 1) & ev; be = (y == 0) & ev
        for mg in (None, 1.5, 3.0, 6.0):
            s = served_sim(t, c, ev, 1e9 if mg is None else mg * base)
            res.append({"seed": seed, "ratio": ratio, "margin": mg, "attack_usd_served": float(c[s & at].sum()), "attack_usd_total": float(c[at].sum()),
                        "attack_served_frac": float(s[at].mean()) if at.any() else None, "benign_served_frac": float(s[be].mean())})
        print("seed", seed, "ratio", ratio, "base", round(base, 5), flush=True)
json.dump(res, open("D:/Cloud RP/extended_evaluation/results/cap_sweep.json", "w"), indent=1)
print("CAP SWEEP DONE")
