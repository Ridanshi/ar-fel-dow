import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
RD = "D:/Cloud RP/extended_evaluation/results"
R_all = {s: json.load(open(f"{RD}/seed{s}.json")) for s in (42, 43, 44, 45, 46) if os.path.exists(f"{RD}/seed{s}.json")}
def complete(r):
    return ('compare' in r and 'noise_aug01' in r['compare'] and 'threat' in r and 'gan_row_conditioned' in r['threat'] and 'blackbox_targeted' in r['threat']
            and 'ipstress' in r and 'benign_overlap' in r['ipstress'] and 'router_tradeoff_clean_test' in r['ipstress'])
seeds = [s for s, r in R_all.items() if complete(r)]
R = {s: R_all[s] for s in seeds}
def ms(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return (float(v.mean()), float(v.std()), len(v)) if len(v) else (None, None, 0)
def fmt(t, pct=True, nd=2):
    m, sd, n = t
    if m is None: return "n/a"
    return (f"{m*100:.{nd}f} ± {sd*100:.{nd}f}" if pct else f"{m:.4f} ± {sd:.4f}")
def have(s, *path):
    d = R[s]
    for p in path:
        if p not in d: return False
        d = d[p]
    return True
agg = {"seeds": seeds}
print("SEEDS:", seeds)
# --- compare
names = ["plain", "pgd_at", "gan_at", "pgd_gan_at", "cost_pgd_gan_at", "noise_aug", "noise_aug01"]
keys = ["clean_acc", "clean_attack_recall", "acc_pgdA", "acc_ganA", "acc_ganB", "acc_pgd_adaptive", "acc_gan_adaptive", "acc_blackbox_all"]
agg["compare"] = {}
print("\nCOMPARE (mean ± std %, n seeds)")
for n in names:
    row = {}
    ns = [s for s in seeds if have(s, "compare", n)]
    for k in keys:
        row[k] = ms([R[s]["compare"][n].get(k) for s in ns])
    row["frg_per_1000"] = ms([R[s]["compare"][n].get("frg_pgdA_per_1000_rows") for s in ns])
    agg["compare"][n] = row
    print(f"{n:16s} n={len(ns)} " + " | ".join(f"{k[:11]}={fmt(row[k])}" for k in keys) + f" | frg/1000rows=${row['frg_per_1000'][0]:.4f}" if ns else n)
# --- threat
print("\nTHREAT")
thr = {}
tkeys = set()
for s in seeds:
    if "threat" in R[s]:
        tkeys |= set(k for k, v in R[s]["threat"].items() if isinstance(v, dict))
for k in sorted(tkeys):
    ns = [s for s in seeds if have(s, "threat", k)]
    sub = set()
    for s in ns:
        v = R[s]["threat"][k]
        if all(isinstance(x, dict) for x in v.values()):
            for lab in v: sub.add(lab)
    if sub:
        thr[k] = {lab: {m: ms([R[s]["threat"][k][lab][m] for s in ns if lab in R[s]["threat"][k]]) for m in ("plain", "pgd_at", "pgd_gan_at")} for lab in sub}
        for lab in sorted(sub):
            print(f"  {k}/{lab}: " + " ".join(f"{m}={fmt(thr[k][lab][m])}" for m in ("plain", "pgd_at", "pgd_gan_at")) + f" (n={len(ns)})")
    else:
        thr[k] = {m: ms([R[s]["threat"][k][m] for s in ns]) for m in ("plain", "pgd_at", "pgd_gan_at")}
        print(f"  {k}: " + " ".join(f"{m}={fmt(thr[k][m])}" for m in ("plain", "pgd_at", "pgd_gan_at")) + f" (n={len(ns)})")
agg["threat"] = thr
# --- ip stress
print("\nCOLD START (pooled over seeds; recall %)")
cs = {}
for b in ("0", "1-2", "3-9", "10-49", "50+"):
    ns = [s for s in seeds if have(s, "ipstress", "coldstart", b)]
    tot = sum(R[s]["ipstress"]["coldstart"][b]["n"] for s in ns)
    row = {"n_pooled": tot}
    for m in ("full", "behavioral_only", "router_thr2", "router_thr9"):
        num = sum((R[s]["ipstress"]["coldstart"][b][m] or 0) * R[s]["ipstress"]["coldstart"][b]["n"] for s in ns)
        row[m] = num / tot if tot else None
    cs[b] = row
    print(f"  seq {b:6s} n={tot:6d} " + " ".join(f"{m}={row[m]*100:.1f}" for m in ("full", "behavioral_only", "router_thr2", "router_thr9")))
agg["coldstart"] = cs
print("\nROTATION (attack recall %)")
rot = {}
for lab in ("seq_0_to_0", "seq_0_to_2", "seq_0_to_9", "seq_0_to_49"):
    ns = [s for s in seeds if have(s, "ipstress", "rotation", lab)]
    rot[lab] = {m: ms([R[s]["ipstress"]["rotation"][lab][m] for s in ns]) for m in ("plain", "pgd_at", "pgd_gan_at", "behavioral_only", "router_thr9")}
    print("  ", lab, " ".join(f"{m}={fmt(rot[lab][m],nd=1)}" for m in rot[lab]), f"(n={len(ns)})")
agg["rotation"] = rot
print("\nBENIGN OVERLAP (benign recall %)")
bo = {}
for lab in ("frac_0.0", "frac_0.01", "frac_0.05", "frac_0.1", "frac_0.25", "frac_0.5", "frac_1.0"):
    ns = [s for s in seeds if have(s, "ipstress", "benign_overlap", lab)]
    bo[lab] = {m: ms([R[s]["ipstress"]["benign_overlap"][lab][m] for s in ns]) for m in ("plain", "pgd_at", "pgd_gan_at", "behavioral_only")}
    print("  ", lab, " ".join(f"{m}={fmt(bo[lab][m],nd=1)}" for m in bo[lab]), f"(n={len(ns)})")
agg["benign_overlap"] = bo
print("\nROUTER TRADEOFF (clean test)")
rt = {}
for m in ("full_model", "behavioral_only", "router_thr2", "router_thr9"):
    ns = [s for s in seeds if have(s, "ipstress", "router_tradeoff_clean_test", m)]
    rt[m] = {k: ms([R[s]["ipstress"]["router_tradeoff_clean_test"][m][k] for s in ns]) for k in ("accuracy", "attack_recall", "benign_recall")}
    print("  ", m, " ".join(f"{k}={fmt(v)}" for k, v in rt[m].items()), f"(n={len(ns)})")
agg["router_tradeoff"] = rt
bh = ms([R[s]["ipstress"]["behavioral_only_overall"]["attack_recall"] for s in seeds if have(s, "ipstress")])
print("behavioral-only overall attack recall", fmt(bh))
# --- fit seconds
print("\nFIT SECONDS (mean over seeds)")
fs = {}
for n in ["plain", "pgd_at", "gan_at", "pgd_gan_at", "cost_pgd_gan_at", "noise_aug", "noise_aug01", "behavioral"]:
    v = [R[s]["fit_seconds"].get(n) for s in seeds if n in R[s].get("fit_seconds", {})]
    fs[n] = ms(v); print("  ", n, f"{fs[n][0]:.0f}s" if fs[n][0] else "n/a", f"(n={fs[n][2]})")
agg["fit_seconds"] = fs
if 42 in R and "overhead" in R[42]:
    agg["overhead_seed42"] = R[42]["overhead"]
json.dump(agg, open(f"{RD}/aggregate.json", "w"), indent=1)
print("\nAGG DONE")
