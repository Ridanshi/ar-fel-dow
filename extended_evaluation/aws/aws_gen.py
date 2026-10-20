"""Drive a real AWS Lambda function with synthetic benign traffic plus simulated bots, via the AWS SDK (restricted profile).
Hard limits: max 8 requests in flight, hard cap on invocations, abort on high error rate or a stop file."""
import sys, os, json, time, base64, re, threading
import numpy as np, boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from synth_benign import benign
seed = int(sys.argv[1]); NB = int(sys.argv[2]) if len(sys.argv) > 2 else 600
MAX_INFLIGHT, HARD_CAP, STOP = 8, 2400, "D:/Cloud RP/extended_evaluation/aws/STOP"
FN, SPAN = "dowlab-work", 330.0
cli = boto3.Session(profile_name="dow-lab", region_name="us-east-1").client("lambda", config=Config(max_pool_connections=16, retries={"max_attempts": 1}, read_timeout=30))
rng = np.random.default_rng(seed)
st, dur, cb = benign(rng, NB, SPAN)
nbots = 20; per = int(NB * 2.34) // nbots
ev = [(float(a), c, float(m), 0) for a, c, m in zip(st, cb, dur)]
for j in range(nbots):
    gap = rng.lognormal(np.log(max(SPAN / (per * 3.0), 0.02)), 0.5)
    s0 = rng.uniform(0, max(SPAN - gap * per, 1.0))
    ts = s0 + np.cumsum(rng.exponential(gap, per)); ts = ts[ts < SPAN]
    for t in ts:
        ev.append((float(t), "bot%02d" % j, float(dur[rng.integers(0, NB)]), 1))
ev.sort()
assert len(ev) <= HARD_CAP, f"plan has {len(ev)} invocations, above hard cap {HARD_CAP}"
print("events", len(ev), "attack share", round(np.mean([e[3] for e in ev]), 3), flush=True)
rows, lock = [], threading.Lock()
inflight = [0]; stats = {"n": 0, "err": 0}
REP = re.compile(r"Duration: ([\d.]+) ms\s+Billed Duration: (\d+) ms\s+Memory Size: (\d+) MB\s+Max Memory Used: (\d+) MB(?:\s+Init Duration: ([\d.]+) ms)?")


def fire(i, e, t0):
    with lock:
        inflight[0] += 1; infl_send = inflight[0]
    ts = time.time() - t0
    r = {"i": i, "client": e[1], "label": e[3], "plan_ms": e[2], "send": ts, "inflight_send": infl_send}
    try:
        resp = cli.invoke(FunctionName=FN, InvocationType="RequestResponse", LogType="Tail", Payload=json.dumps({"client": e[1], "ms": int(e[2])}))
        resp["Payload"].read()
        tr = time.time() - t0
        m = REP.search(base64.b64decode(resp.get("LogResult", "")).decode("utf8", "ignore"))
        r.update({"recv": tr, "status": "error" if resp.get("FunctionError") else "success"})
        if m:
            r.update({"duration_ms": float(m.group(1)), "billed_ms": int(m.group(2)), "mem_used_mb": int(m.group(4)), "init_ms": float(m.group(5) or 0)})
    except Exception as ex:
        r.update({"recv": time.time() - t0, "status": "exc:" + type(ex).__name__})
    with lock:
        inflight[0] -= 1; r["inflight_recv"] = inflight[0]; rows.append(r)
        stats["n"] += 1; stats["err"] += r["status"] != "success"


t0 = time.time(); aborted = None
with ThreadPoolExecutor(MAX_INFLIGHT) as pool:
    for i, e in enumerate(ev):
        d = t0 + e[0] - time.time()
        if d > 0: time.sleep(d)
        while inflight[0] >= MAX_INFLIGHT: time.sleep(0.005)
        if os.path.exists(STOP): aborted = "stop file"; break
        if stats["n"] >= 200 and stats["err"] / stats["n"] > 0.2: aborted = "error rate > 20%"; break
        pool.submit(fire, i, e, t0)
print("done sending", round(time.time() - t0, 1), "s; completed", stats["n"], "errors", stats["err"], "aborted:", aborted, flush=True)
json.dump({"seed": seed, "n_events": len(ev), "aborted": aborted, "rows": sorted(rows, key=lambda r: r["i"])}, open(f"D:/Cloud RP/extended_evaluation/aws/aws_s{seed}.json", "w"))
print("saved", len(rows), "rows")
