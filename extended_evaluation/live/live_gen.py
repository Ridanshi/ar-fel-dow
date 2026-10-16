"""Drive a real OpenWhisk platform (local VM) with synthetic benign traffic plus simulated bots,
then pull the platform's own activation records. Client identity is a request parameter (as an API key would be)."""
import sys, json, time, base64, threading, urllib.request, urllib.error
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
seed = int(sys.argv[1]); NB = int(sys.argv[2]) if len(sys.argv) > 2 else 1800; COMP = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0; CAP_MS = 1000
KEY = "23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP"
H = {"Authorization": "Basic " + base64.b64encode(KEY.encode()).decode(), "Content-Type": "application/json"}
BASE = "http://127.0.0.1:3233/api/v1/namespaces/_"


def call(method, path, body=None, timeout=60):
    r = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), headers=H, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read() or b"null")


call("PUT", "/actions/work?overwrite=true", {"exec": {"kind": "nodejs:20", "code": "function main(p){return new Promise(r=>setTimeout(()=>r({ok:1,c:p.client}),p.ms));}"}, "limits": {"timeout": 30000, "memory": 256}})
rng = np.random.default_rng(seed)
from synth_benign import benign
span = 330.0
st, dur, cli_b = benign(rng, NB, span)
nbots = 20; na = int(NB * 2.34); per = na // nbots
ev = [(float(s_), c, float(m), 0) for s_, c, m in zip(st, cli_b, dur)]
for j in range(nbots):
    gap = rng.lognormal(np.log(max(span / (per * 3.0), 0.02)), 0.5)
    s0 = rng.uniform(0, max(span - gap * per, 1.0))
    ts = s0 + np.cumsum(rng.exponential(gap, per)); ts = ts[ts < span]
    for t in ts:
        ev.append((float(t), "bot%02d" % j, float(dur[rng.integers(0, NB)]), 1))
ev.sort()
print("events", len(ev), "span s", round(span, 1), "attack share", round(np.mean([e[3] for e in ev]), 3), flush=True)
sent = {}
errs = {}
lock = threading.Lock()


def fire(i, e):
    t = time.time()
    try:
        r = call("POST", "/actions/work?blocking=false", {"client": e[1], "ms": int(e[2])}, timeout=30)
        with lock:
            sent[r["activationId"]] = (i, t)
    except Exception as ex:
        with lock:
            sent["ERR%d" % i] = (i, t)
            errs[type(ex).__name__ + (str(getattr(ex, "code", "")))] = errs.get(type(ex).__name__ + (str(getattr(ex, "code", ""))), 0) + 1


t0 = time.time(); t0ms = int(t0 * 1000) - 5000
with ThreadPoolExecutor(64) as pool:
    for i, e in enumerate(ev):
        dly = t0 + e[0] - time.time()
        if dly > 0:
            time.sleep(dly)
        pool.submit(fire, i, e)
print("all sent", round(time.time() - t0, 1), "s; errors", sum(k.startswith("ERR") for k in sent), errs, flush=True)
json.dump(sent, open(f"D:/Cloud RP/extended_evaluation/live/sent_s{seed}.json", "w"))
nok = sum(not k.startswith("ERR") for k in sent)
acts = {}
last = -1; stable = 0
for rnd in range(60):
    time.sleep(20)
    acts = {}
    skip = 0
    while True:
        page = call("GET", f"/activations?limit=200&skip={skip}&docs=true&since={t0ms}&name=work", timeout=120)
        for a in page:
            acts[a["activationId"]] = a
        if len(page) < 200:
            break
        skip += 200
    got = sum(1 for k in sent if k in acts)
    print("poll", rnd, "completed", got, "of", nok, flush=True)
    if got >= nok:
        break
    stable = stable + 1 if got == last else 0
    last = got
    if stable >= 4:
        break
rows = []
for aid, (i, ts) in sent.items():
    a = acts.get(aid)
    if a is None:
        continue
    ann = {x["key"]: x["value"] for x in a.get("annotations", [])}
    e = ev[i]
    rows.append({"i": i, "client": e[1], "label": e[3], "plan_ms": e[2], "send": ts - t0, "start": a["start"] / 1000.0 - t0, "end": a["end"] / 1000.0 - t0,
                 "duration_ms": a["duration"], "wait_ms": ann.get("waitTime", 0), "init_ms": ann.get("initTime", 0), "status": a["response"]["status"]})
json.dump({"seed": seed, "n_events": len(ev), "n_rows": len(rows), "rows": rows}, open(f"D:/Cloud RP/extended_evaluation/live/live_s{seed}.json", "w"))
print("collected", len(rows), "of", len(ev), flush=True)
