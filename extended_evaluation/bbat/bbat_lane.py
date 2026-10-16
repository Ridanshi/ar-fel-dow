import sys, json, os
sys.path.insert(0, "D:/Cloud RP/extended_evaluation/live"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation/bbat")
which = sys.argv[1]
from arfel_dow.evaluate import _train_test_split
log = lambda m: print(m, flush=True)
def save(path, out): json.dump(out, open(path, "w"), indent=1)
if which in ("eps_live", "bbat_bccc", "eps_bccc"):
    import numpy as np
    if which == "eps_live":
        from live_eval import features
        from eps_core import run_eps
        runs = [42, 43, 44]; D = {r: features(r) for r in runs}
        Xp = np.vstack([D[r][0] for r in runs]); yp = np.concatenate([D[r][1] for r in runs]); cp = np.concatenate([[f"{r}:{c}" for c in D[r][2]] for r in runs])
        P = "D:/Cloud RP/extended_evaluation/results/eps_live.json"; out = json.load(open(P)) if os.path.exists(P) else {}
        for s in (1, 2, 3, 4, 5):
            if f"split{s}" in out: continue
            Xtr, ytr, Xte, yte, _, _ = _train_test_split(Xp, yp, 0.3, s, cp)
            out[f"split{s}"] = run_eps(100 + s, f"live_pool{s}", f"live_pool{s}_plain", Xtr, ytr, Xte, yte, log); save(P, out)
    else:
        from sklearn.feature_selection import SelectKBest, f_classif
        Z = np.load("D:/Cloud RP/extended_evaluation/bccc/bccc_data.npz", allow_pickle=True); X0, y, cnt, cli = Z["X"], Z["y"], Z["cnt"], Z["cli"]
        P = f"D:/Cloud RP/extended_evaluation/results/{which}.json"; out = json.load(open(P)) if os.path.exists(P) else {}
        for s in (1, 2, 3, 4, 5):
            if f"split{s}" in out: continue
            Xa = np.column_stack([cnt, X0]); _, _, _, _, tr, te = _train_test_split(Xa, y, 0.3, s, cli)
            sel = SelectKBest(f_classif, k=40).fit(X0[tr], y[tr]).get_support(indices=True)
            mk = lambda ix: np.column_stack([X0[ix][:, sel[0]], cnt[ix], X0[ix][:, sel[1:]]]).astype(float)
            if which == "bbat_bccc":
                from bbat_core import run_bbat
                out[f"split{s}"] = run_bbat(300 + s, f"live_bccc{s}", f"live_bccc{s}_plain", mk(tr), y[tr], mk(te), y[te], log)
            else:
                from eps_core import run_eps
                out[f"split{s}"] = run_eps(300 + s, f"live_bccc{s}", f"live_bccc{s}_plain", mk(tr), y[tr], mk(te), y[te], log)
            save(P, out)
print("LANE DONE", which)
