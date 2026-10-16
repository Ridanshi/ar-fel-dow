"""Translate detection failures into dollars under explicit pricing assumptions."""
import sys, json
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from rlib import *
from run_experiment import load_and_prepare
from arfel_dow.evaluate import _train_test_split

seed = int(sys.argv[1])
X, y, dur, mem, groups, cols = load_and_prepare()
Xtr, ytr, Xte, yte, tr_idx, te_idx = _train_test_split(X, y, 0.3, seed, groups)
scaler = FeatureScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
lo, hi = Xtr_s.min(0), Xtr_s.max(0)
d = X.shape[1]
IP = cols.index("ip_request_sequence_number")
at = yte == 1
Xat = Xte[at]; yat = yte[at]; Xat_s = scaler.transform(Xat)
dur_at = dur[te_idx][at]; mem_at = mem[te_idx][at]
models = {n: pickle.load(open(f"{CACHE}/s{seed}_{n}.pkl", "rb"))["model"] for n in ("plain", "pgd_at", "pgd_gan_at")}
sur0 = Surrogate(input_dim=d, random_state=seed).fit(Xtr_s, models["plain"].predict_proba(Xtr), epochs=200)
Xpgd = scaler.inverse_transform(pgd_attack(sur0, Xat_s, yat, EPS, EPS / 8, 15))
mip = np.zeros(d, dtype=np.float32); mip[IP] = 1.0
sub = np.random.default_rng(seed + 77).choice(len(Xat), size=min(3000, len(Xat)), replace=False)
bb = {}
for n, m in models.items():
    adv, _ = blackbox_search(m, Xat_s[sub], scaler, EPS, mip, [], lo, hi, 30, np.random.default_rng(seed + 8))
    bb[n] = scaler.inverse_transform(adv)

assump = {
    "paper_vm_memory_bucket": mem_at,
    "capped_at_lambda_max_10240MB": np.minimum(mem_at, 10240.0),
    "fixed_1024MB": np.full_like(mem_at, 1024.0),
}
hours = 14.0 + 59.0 / 60.0 + 5.0 / 3600.0  # attack-traffic window in DoWTS (15.0 h)
out = {"n_attack_test_rows": int(len(Xat)), "attack_requests_per_hour_in_dataset": 131072 / hours, "assumptions": {}}
for aname, memv in assump.items():
    c = invocation_costs(dur_at, memv)
    rec = {"mean_cost_per_attack_invocation_usd": float(c.mean()), "scenarios": {}}
    for sname in ("pgdA_transfer_unconstrained", "ip_counter_only_blackbox"):
        rec["scenarios"][sname] = {}
        for n, m in models.items():
            pc = m.predict(Xat)
            if sname.startswith("pgd"):
                pa = m.predict(Xpgd); cc = c
                miss_clean = float((pc == 0).mean()); miss_adv = float((pa == 0).mean())
                gap = float(cc[pa == 0].sum() - cc[pc == 0].sum()) / len(Xat)
            else:
                pa = m.predict(bb[n]); cs = c[sub]
                miss_clean = float((pc[sub] == 0).mean()); miss_adv = float((pa == 0).mean())
                gap = float(cs[pa == 0].sum() - cs[pc[sub] == 0].sum()) / len(sub)
            rec["scenarios"][sname][n] = {"miss_rate_clean": miss_clean, "miss_rate_adversarial": miss_adv,
                                          "extra_usd_per_attack_invocation": gap,
                                          "extra_usd_per_1M_attack_invocations": gap * 1e6,
                                          "extra_usd_per_hour_at_dataset_attack_rate": gap * out["attack_requests_per_hour_in_dataset"],
                                          "extra_usd_per_hour_at_100x_dataset_attack_rate": gap * out["attack_requests_per_hour_in_dataset"] * 100}
    out["assumptions"][aname] = rec
json.dump(out, open(f"D:/Cloud RP/extended_evaluation/results/seed{seed}_frg_scaling.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("FRG DONE")
