"""Validity of the paper's adversarial examples and accuracy on valid vs invalid rows."""
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
at_te = yte == 1
Xat = Xte[at_te]; yat = yte[at_te]; Xat_s = scaler.transform(Xat)
models = {}
for n in ("plain", "pgd_at", "pgd_gan_at"):
    models[n] = pickle.load(open(f"{CACHE}/s{seed}_{n}.pkl", "rb"))["model"]
sur0 = Surrogate(input_dim=d, random_state=seed).fit(Xtr_s, models["plain"].predict_proba(Xtr), epochs=200)
Xb_s = scaler.transform(Xtr[ytr == 0]); Xa_s = scaler.transform(Xtr[ytr == 1])
Xpgd_s = pgd_attack(sur0, Xat_s, yat, EPS, EPS / 8, 15)
ganA = AdaptiveGANAttacker(input_dim=d, noise_dim=8, random_state=seed).fit(Xb_s, Xa_s, sur0, epochs=100)
Xgan_s = ganA.generate(len(Xat))
tol = 1e-6
out = {"n_attack_rows": int(len(Xat))}
for label, Zs in (("pgdA_unconstrained", Xpgd_s), ("ganA", Xgan_s)):
    viol = (Zs < lo - tol) | (Zs > hi + tol)
    valid = ~viol.any(axis=1)
    Z = scaler.inverse_transform(Zs)
    rec = {"fraction_rows_fully_valid": float(valid.mean()),
           "per_feature_violation_rate": {c: float(viol[:, i].mean()) for i, c in enumerate(cols) if viol[:, i].mean() > 0.01}}
    for n, m in models.items():
        pr = m.predict(Z)
        rec["acc_" + n + "_all_rows"] = float((pr == 1).mean())
        rec["acc_" + n + "_valid_rows"] = float((pr[valid] == 1).mean()) if valid.any() else None
        rec["acc_" + n + "_invalid_rows"] = float((pr[~valid] == 1).mean()) if (~valid).any() else None
    out[label] = rec
# range-clipped (valid-range) PGD, unrestricted otherwise
Zc = scaler.inverse_transform(masked_pgd(sur0, Xat_s, yat, EPS, EPS / 8, 15, np.ones(d, dtype=np.float32), [], lo, hi))
out["pgd_range_clipped"] = {n: float((m.predict(Zc) == 1).mean()) for n, m in models.items()}
# which perturbation does the unconstrained attack apply to the IP-frequency feature?
IPi = cols.index("ip_request_sequence_number")
out["ip_feature"] = {"std_requests_(1 standardized unit)": float(scaler._std[IPi]),
                     "mean_shift_pgd_unconstrained_in_requests": float((scaler.inverse_transform(Xpgd_s)[:, IPi] - Xat[:, IPi]).mean())}
json.dump(out, open(f"D:/Cloud RP/extended_evaluation/results/seed{seed}_validity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("VALIDITY DONE")
