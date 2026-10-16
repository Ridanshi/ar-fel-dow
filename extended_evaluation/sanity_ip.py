import sys, json
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from rlib import *
from run_experiment import load_and_prepare
from arfel_dow.evaluate import _train_test_split
X, y, dur, mem, groups, cols = load_and_prepare()
IP = cols.index("ip_request_sequence_number")
for seed in [int(a) for a in sys.argv[1:]]:
    Xtr, ytr, Xte, yte, tr, te = _train_test_split(X, y, 0.3, seed, groups)
    sd = float(Xtr[:, IP].std())
    seq = Xte[yte == 1][:, IP]
    plain = pickle.load(open(f"{CACHE}/s{seed}_plain.pkl", "rb"))["model"]
    probe = Xte[yte == 1][:2000].copy()
    thr = None
    for v in (0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50):
        p2 = probe.copy()
        p2[:, IP] = v
        if thr is None and (plain.predict(p2) == 1).mean() > 0.5:
            thr = v
    share = float((seq <= sd + thr).mean())
    out = {"seed": seed, "std_requests": sd, "detection_count": thr, "share_attack_rows_within_one_unit": share, "predicted_accuracy": 1 - share}
    json.dump(out, open(f"D:/Cloud RP/extended_evaluation/results/seed{seed}_sanity.json", "w"), indent=1)
    print(out)
print("SANITY DONE")
