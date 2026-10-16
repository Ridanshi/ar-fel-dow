"""BCCC-cPacket-Cloud-DDoS-2024 (Monday+Tuesday): Benign vs Attack flows, causal per-source-IP count, stratified subsample.
Two passes to stay within memory: (1) ids/time/label -> counts and sample; (2) features for sampled rows only."""
import numpy as np, pandas as pd
D = "D:/Cloud RP/ext_data/bccc_cloud_ddos/out/CSVs/"
DAYS = ("Monday_18_Dec_2023", "Tuesday_19_Dec_2023")
META = ["flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "label"]
ids = []
for day in DAYS:
    m = pd.read_csv(D + day + ".csv", usecols=["src_ip", "timestamp", "label"])
    m["day"] = day; m["row"] = np.arange(len(m)); ids.append(m[m.label.isin(["Benign", "Attack"])])
m = pd.concat(ids, ignore_index=True)
m["ts"] = pd.to_datetime(m["timestamp"], errors="coerce")
m = m.sort_values("ts", kind="stable").reset_index(drop=True)
m["cnt"] = m.groupby("src_ip").cumcount()
print("rows", len(m), m.label.value_counts().to_dict(), "ips", m.src_ip.nunique(), flush=True)
rng = np.random.default_rng(0)
CAP = 400
m["r"] = rng.random(len(m))
m["rk"] = m.groupby("src_ip")["r"].rank(method="first")
m = m[m.rk <= CAP].drop(columns=["r", "rk"])
print("after per-IP cap", len(m), m.label.value_counts().to_dict(), flush=True)
keep = m.iloc[np.sort(rng.choice(len(m), size=min(100000, len(m)), replace=False))].copy()
parts = []
for day in DAYS:
    want = set(keep.loc[keep.day == day, "row"])
    off = 0
    for ch in pd.read_csv(D + day + ".csv", chunksize=40000, low_memory=False):
        ch = ch.drop(columns=[c for c in ("flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "label") if c in ch.columns])
        idx = np.arange(off, off + len(ch)); off += len(ch)
        sel = np.isin(idx, list(want)) if len(want) < 1 else np.isin(idx, np.fromiter(want, dtype=np.int64))
        c2 = ch[sel].select_dtypes(include=[np.number]).astype(np.float32)
        c2["row"] = idx[sel]; c2["day"] = day
        parts.append(c2)
    print(day, "features read", flush=True)
F = pd.concat(parts, ignore_index=True)
K = keep.merge(F, on=["day", "row"], how="inner")
y = (K.label == "Attack").astype(int).to_numpy()
num = K.drop(columns=["src_ip", "timestamp", "label", "day", "row", "ts", "cnt"]).select_dtypes(include=[np.number])
num = num.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
num = num.loc[:, num.std() > 0]
print("features", num.shape[1], "rows", len(K), "attack share", y.mean().round(3))
np.savez_compressed("D:/Cloud RP/extended_evaluation/bccc/bccc_data.npz", X=num.to_numpy(), y=y, cnt=K["cnt"].to_numpy().astype(np.float32),
                    cli=K["src_ip"].to_numpy().astype(str), cols=np.array(num.columns))
att_ips = set(K.src_ip[y == 1]); ben_ips = set(K.src_ip[y == 0])
print("saved; attack ips", len(att_ips), "benign ips", len(ben_ips), "mixed", len(att_ips & ben_ips))
