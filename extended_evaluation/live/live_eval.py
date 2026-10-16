"""Evaluate the detector and attacks on live OpenWhisk telemetry (platform-measured start/end/duration/wait/init)."""
import sys, json, os
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from rlib import *
from arfel_dow.evaluate import _train_test_split
COLS = ["SubmitTime", "ip_request_sequence_number", "client_gap", "FunctionDuration", "InvocationDelay", "ColdStart", "ActiveFunctionsAtRequest", "ActiveFunctionsAtResponse"]
CNT = 1


def features(run):
    R = json.load(open(f"D:/Cloud RP/extended_evaluation/live/live_s{run}.json"))["rows"]
    R = sorted(R, key=lambda r: r["send"])
    send = np.array([r["send"] for r in R]); st = np.array([r["start"] for r in R]); en = np.array([r["end"] for r in R])
    cli = np.array([r["client"] for r in R]); y = np.array([r["label"] for r in R])
    dur = np.array([r["duration_ms"] for r in R], float); wait = np.array([r["wait_ms"] for r in R], float); ini = np.array([r["init_ms"] > 0 for r in R], float)
    es = np.sort(en); ss = np.sort(st)
    act_req = np.searchsorted(ss, send, side="right") - np.searchsorted(es, send, side="right")
    act_resp = np.searchsorted(ss, en, side="right") - np.searchsorted(es, en, side="left")
    cnt = np.zeros(len(R)); gap = np.zeros(len(R)); last = {}; seen = {}
    span = send.max()
    for i, c in enumerate(cli):
        cnt[i] = seen.get(c, 0); seen[c] = cnt[i] + 1
        gap[i] = send[i] - last[c] if c in last else span
        last[c] = send[i]
    X = np.column_stack([send, cnt, gap, dur, wait, ini, act_req, act_resp]).astype(float)
    return X, y, cli


def evaluate(run, Xtr, ytr, Xte, yte, tag, log=print):
    seed = run
    sc = FeatureScaler().fit(Xtr); Xtr_s = sc.transform(Xtr); lo = Xtr_s.min(0); hi = Xtr_s.max(0)
    w = balanced_sample_weight(ytr)
    plain, _ = cached_fit(f"live_{tag}_plain", seed, Xtr, ytr, w, log)
    nocnt, _ = cached_fit(f"live_{tag}_nocnt", seed, np.delete(Xtr, CNT, 1), ytr, w, log)
    at_tr = ytr == 1
    sur = Surrogate(input_dim=Xtr.shape[1], random_state=seed).fit(Xtr_s, plain.predict_proba(Xtr), epochs=200)
    Xadv_tr = sc.inverse_transform(pgd_attack(sur, Xtr_s[at_tr], np.ones(at_tr.sum(), dtype=np.int64), EPS, EPS / 8, 15))
    Xa = np.vstack([Xtr, Xadv_tr]); ya = np.concatenate([ytr, np.ones(len(Xadv_tr), dtype=np.int64)])
    pgdat, _ = cached_fit(f"live_{tag}_pgdat", seed, Xa, ya, balanced_sample_weight(ya), log)
    at_te = yte == 1; Xat = Xte[at_te]; Xat_s = sc.transform(Xat)
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "n_attack_test": int(at_te.sum()), "n_benign_test": int((~at_te).sum())}
    yp = nocnt.predict(np.delete(Xte, CNT, 1)); out["nocount_acc"] = float((yp == yte).mean())
    Xpgd = sc.inverse_transform(pgd_attack(sur, Xat_s, np.ones(len(Xat), dtype=np.int64), EPS, EPS / 8, 15))
    nf = Xtr.shape[1]
    def mk(ix):
        v = np.zeros(nf, dtype=np.float32); v[ix] = 1.0; return v
    masks = {"all": mk(list(range(nf))), "no_count": mk([i for i in range(nf) if i != CNT]), "count_only": mk([CNT])}
    sub = np.random.default_rng(seed + 9).choice(len(Xat), size=min(1500, len(Xat)), replace=False)
    for name, mod in (("plain", plain), ("pgd_at", pgdat)):
        p = mod.predict(Xte); r = {"acc": float((p == yte).mean()), "benign_recall": float((p[~at_te] == 0).mean()), "attack_recall": float((p[at_te] == 1).mean())}
        r["pgd_transfer"] = float((mod.predict(Xpgd) == 1).mean())
        for mn, mv in masks.items():
            adv, q = blackbox_search(mod, Xat_s[sub], sc, EPS, mv, [], lo, hi, 30, np.random.default_rng(seed + 5))
            r["bb_" + mn] = float((mod.predict(sc.inverse_transform(adv)) == 1).mean())
        for lab, hc in (("0_2", 2), ("0_9", 9)):
            Z = Xat.copy(); Z[:, CNT] = np.random.default_rng(seed + 11).integers(0, hc + 1, len(Z)); r["rot_" + lab] = float((mod.predict(Z) == 1).mean())
        out[name] = r
        log(f"{tag} {name}: acc {r['acc']:.3f} benrec {r['benign_recall']:.3f} pgd {r['pgd_transfer']:.3f} bb_all {r['bb_all']:.3f} bb_nocnt {r['bb_no_count']:.3f} rot02 {r['rot_0_2']:.3f}")
    return out, plain, pgdat


if __name__ == "__main__":
    runs = [int(a) for a in sys.argv[1:]]
    res = {}
    data = {r: features(r) for r in runs}
    for r in runs:
        X, y, cli = data[r]
        Xtr, ytr, Xte, yte, tr, te = _train_test_split(X, y, 0.3, r, cli)
        res[f"within_{r}"], plain, pgdat = evaluate(r, Xtr, ytr, Xte, yte, f"w{r}")
        for o in runs:
            if o != r:
                Xo, yo, _ = data[o]
                res.setdefault(f"cross_{r}_to_{o}", {"plain_acc": float((plain.predict(Xo) == yo).mean()), "plain_benign_recall": float((plain.predict(Xo[yo == 0]) == 0).mean()),
                                                      "plain_attack_recall": float((plain.predict(Xo[yo == 1]) == 1).mean()), "pgdat_acc": float((pgdat.predict(Xo) == yo).mean())})
    json.dump(res, open("D:/Cloud RP/extended_evaluation/results/live_eval.json", "w"), indent=1)
    print("LIVE EVAL DONE")
