"""Evaluate the detector and attacks on telemetry from real AWS Lambda invocations (pooled runs, client-grouped splits)."""
import sys, json
sys.path.insert(0, "D:/Cloud RP/extended_evaluation/live"); sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
import numpy as np
import live_eval
from live_eval import evaluate
from arfel_dow.evaluate import _train_test_split
live_eval.CNT = 1


def features(run):
    R = [r for r in json.load(open(f"D:/Cloud RP/extended_evaluation/aws/aws_s{run}.json"))["rows"] if r["status"] == "success" and "duration_ms" in r]
    R.sort(key=lambda r: r["send"])
    send = np.array([r["send"] for r in R]); dur = np.array([r["duration_ms"] for r in R]); bil = np.array([r["billed_ms"] for r in R], float)
    cold = np.array([float(r.get("init_ms", 0) > 0) for r in R]); delay = np.array([(r["recv"] - r["send"]) * 1000 - r["duration_ms"] for r in R])
    cli = np.array([r["client"] for r in R]); y = np.array([r["label"] for r in R])
    isn = np.array([r["inflight_send"] for r in R], float); isr = np.array([r["inflight_recv"] for r in R], float)
    cnt = np.zeros(len(R)); gap = np.zeros(len(R)); last = {}; seen = {}; span = send.max()
    for i, c in enumerate(cli):
        cnt[i] = seen.get(c, 0); seen[c] = cnt[i] + 1
        gap[i] = send[i] - last[c] if c in last else span
        last[c] = send[i]
    return np.column_stack([send, cnt, gap, dur, delay, cold, isn, isr]).astype(float), y, cli


if __name__ == "__main__":
    runs = [42, 43, 44]
    D = {r: features(r) for r in runs}
    Xp = np.vstack([D[r][0] for r in runs]); yp = np.concatenate([D[r][1] for r in runs])
    cp = np.concatenate([[f"{r}:{c}" for c in D[r][2]] for r in runs])
    out = {"pooled": {}, "loro": {}}
    for s in (1, 2, 3, 4, 5):
        Xtr, ytr, Xte, yte, _, _ = _train_test_split(Xp, yp, 0.3, s, cp)
        out["pooled"][f"split{s}"], _, _ = evaluate(400 + s, Xtr, ytr, Xte, yte, f"aws_pool{s}")
    rid = np.concatenate([[r] * len(D[r][1]) for r in runs])
    for r in runs:
        out["loro"][f"holdout{r}"], _, _ = evaluate(500 + r, Xp[rid != r], yp[rid != r], Xp[rid == r], yp[rid == r], f"aws_loro{r}")
    json.dump(out, open("D:/Cloud RP/extended_evaluation/results/aws_pool.json", "w"), indent=1)
    print("AWS EVAL DONE")
