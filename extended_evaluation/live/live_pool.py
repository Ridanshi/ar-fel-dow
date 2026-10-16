import sys, json
sys.path.insert(0, "D:/Cloud RP/extended_evaluation/live"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from live_eval import *
from arfel_dow.evaluate import _train_test_split
runs = [42, 43, 44]
D = {r: features(r) for r in runs}
Xp = np.vstack([D[r][0] for r in runs]); yp = np.concatenate([D[r][1] for r in runs])
cp = np.concatenate([[f"{r}:{c}" for c in D[r][2]] for r in runs])
rid = np.concatenate([[r] * len(D[r][1]) for r in runs])
out = {"pooled": {}, "loro": {}}
for s in (1, 2, 3, 4, 5):
    Xtr, ytr, Xte, yte, tr, te = _train_test_split(Xp, yp, 0.3, s, cp)
    out["pooled"][f"split{s}"], _, _ = evaluate(100 + s, Xtr, ytr, Xte, yte, f"pool{s}")
for r in runs:
    tr, te = rid != r, rid == r
    out["loro"][f"holdout{r}"], _, _ = evaluate(200 + r, Xp[tr], yp[tr], Xp[te], yp[te], f"loro{r}")
json.dump(out, open("D:/Cloud RP/extended_evaluation/results/live_pool.json", "w"), indent=1)
print("POOL DONE")
