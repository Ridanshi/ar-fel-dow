import sys, json, os
sys.path.insert(0, "D:/Cloud RP/extended_evaluation"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation/bbat"); sys.path.insert(0, "D:/Cloud RP/scripts")
from bbat_core import *
from run_experiment import load_and_prepare
from arfel_dow.evaluate import _train_test_split
X, y, dur, mem, groups, cols = load_and_prepare()
P = "D:/Cloud RP/extended_evaluation/results/bbat_dowts.json"
out = json.load(open(P)) if os.path.exists(P) else {}
for seed in (42, 43, 44):
    if f"seed{seed}" in out: continue
    Xtr, ytr, Xte, yte, _, _ = _train_test_split(X, y, 0.3, seed, groups)
    out[f"seed{seed}"] = run_bbat(seed, f"s{seed}dowts", "plain", Xtr, ytr, Xte, yte, lambda m: print(m, flush=True))
    json.dump(out, open(P, "w"), indent=1)
print("LANE DONE dowts")
