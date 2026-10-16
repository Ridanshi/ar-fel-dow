import sys, json, os
sys.path.insert(0, "D:/Cloud RP/extended_evaluation/live"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation/bbat")
from live_eval import features
from bbat_core import *
from arfel_dow.evaluate import _train_test_split
runs = [42, 43, 44]
D = {r: features(r) for r in runs}
Xp = np.vstack([D[r][0] for r in runs]); yp = np.concatenate([D[r][1] for r in runs])
cp = np.concatenate([[f"{r}:{c}" for c in D[r][2]] for r in runs])
OUTP = "D:/Cloud RP/extended_evaluation/results/bbat_live.json"
out = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
for s in (1, 2, 3, 4, 5):
    if f"split{s}" in out: continue
    Xtr, ytr, Xte, yte, tr, te = _train_test_split(Xp, yp, 0.3, s, cp)
    out[f"split{s}"] = run_bbat(100 + s, f"live_pool{s}", f"live_pool{s}_plain", Xtr, ytr, Xte, yte, lambda m: print(m, flush=True))
    json.dump(out, open(OUTP, "w"), indent=1)
print("BBAT LIVE DONE")
