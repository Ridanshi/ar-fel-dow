import sys, json
sys.path.insert(0, "D:/Cloud RP/extended_evaluation/live"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
import live_eval
from live_eval import *
from arfel_dow.evaluate import _train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
Z = np.load("D:/Cloud RP/extended_evaluation/bccc/bccc_data.npz", allow_pickle=True)
X0, y, cnt, cli = Z["X"], Z["y"], Z["cnt"], Z["cli"]
live_eval.CNT = 1
import os
OUTP = "D:/Cloud RP/extended_evaluation/results/bccc_eval.json"
out = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
for s in (1, 2, 3, 4, 5):
    if f"split{s}" in out:
        continue
    idx = np.arange(len(y))
    Xa = np.column_stack([cnt, X0])
    _, _, _, _, tr, te = _train_test_split(Xa, y, 0.3, s, cli)
    sk = SelectKBest(f_classif, k=40).fit(X0[tr], y[tr])
    sel = sk.get_support(indices=True)
    def mk(ix):
        return np.column_stack([X0[ix][:, sel[0]], cnt[ix], X0[ix][:, sel[1:]]]).astype(float)
    Xtr, Xte = mk(tr), mk(te)
    print("split", s, "train", len(tr), "test", len(te), "test attack share", y[te].mean().round(3), "features", Xtr.shape[1], flush=True)
    out[f"split{s}"], _, _ = live_eval.evaluate(300 + s, Xtr, y[tr], Xte, y[te], f"bccc{s}")
    json.dump(out, open("D:/Cloud RP/extended_evaluation/results/bccc_eval.json", "w"), indent=1)
print("BCCC EVAL DONE")
