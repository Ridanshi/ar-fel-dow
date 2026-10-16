"""Extended-evaluation experiment suite for one seed. Resumable: models are cached to disk and
each stage's results are written to results/seed{N}.json as soon as they finish."""
import sys, json, os, time, threading, argparse
sys.path.insert(0, "D:/Cloud RP/extended_evaluation")
from rlib import *
from run_experiment import load_and_prepare
from arfel_dow.federated import FederatedEnsembleTrainer
from arfel_dow.evaluate import _partition_for_clients, _train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--stages", default="fit,compare,threat,ipstress,overhead")
ap.add_argument("--neval_smooth", type=int, default=10000)
args = ap.parse_args()
seed = args.seed
stages = args.stages.split(",")
OUT = f"D:/Cloud RP/extended_evaluation/results/seed{seed}.json"
R = json.load(open(OUT)) if os.path.exists(OUT) else {}


def log(m):
    print(time.strftime("%H:%M:%S"), f"[s{seed}]", m, flush=True)


def save():
    json.dump(R, open(OUT, "w"), indent=1)


# ------------------------------------------------------------------ data
X, y, dur, mem, groups, cols = load_and_prepare()
Xtr, ytr, Xte, yte, tr_idx, te_idx = _train_test_split(X, y, 0.3, seed, groups)
costs_all = invocation_costs(dur, mem)
d = X.shape[1]
idx, _, _, _ = feature_groups(cols)
IP = idx["ip_request_sequence_number"]
masks, onedir = make_masks(cols)
scaler = FeatureScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
lo = Xtr_s.min(axis=0)
hi = Xtr_s.max(axis=0)
at_tr = ytr == 1
Xat_tr = Xtr[at_tr]
at_te = yte == 1
Xat = Xte[at_te]
yat = yte[at_te]
cost_at = costs_all[te_idx][at_te]
Xat_s = scaler.transform(Xat)
Xb_s = scaler.transform(Xtr[ytr == 0])
Xa_s = scaler.transform(Xat_tr)
R["meta"] = {"n_features": int(d), "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "n_attack_test": int(at_te.sum()),
             "n_benign_test": int((~at_te).sum()), "mean_cost_attack_row_usd": float(cost_at.mean()),
             "feature_columns": cols}
save()
log(f"data ok: features={d}, train={len(Xtr)}, test={len(Xte)}, attack_test={int(at_te.sum())}")

# ------------------------------------------------------------------ fits
w_plain = balanced_sample_weight(ytr)
plain, sec = cached_fit("plain", seed, Xtr, ytr, w_plain, log)
R.setdefault("fit_seconds", {})["plain"] = sec
sur0 = Surrogate(input_dim=d, random_state=seed).fit(Xtr_s, plain.predict_proba(Xtr), epochs=200)
ones_at = np.ones(len(Xat_tr), dtype=np.int64)
Xadv_pgd_tr = scaler.inverse_transform(pgd_attack(sur0, scaler.transform(Xat_tr), ones_at, EPS, EPS / 8, 15))
gan_tr = AdaptiveGANAttacker(input_dim=d, noise_dim=8, random_state=seed + 500).fit(Xb_s, Xa_s, sur0, epochs=100)
Xgan_tr = scaler.inverse_transform(gan_tr.generate(len(Xat_tr)))
ones_aug = ones_at
mult = costs_all[tr_idx][at_tr]
mult = mult / mult.mean()


def aug_fit(name, parts, ys, wmult=None):
    Xa = np.vstack(parts)
    ya = np.concatenate(ys)
    w = balanced_sample_weight(ya)
    if wmult is not None:
        w = w * wmult
    return cached_fit(name, seed, Xa, ya, w, log)


models = {"plain": plain}
m, s = aug_fit("pgd_at", [Xtr, Xadv_pgd_tr], [ytr, ones_aug])
models["pgd_at"] = m
R["fit_seconds"]["pgd_at"] = s
m, s = aug_fit("gan_at", [Xtr, Xgan_tr], [ytr, ones_aug])
models["gan_at"] = m
R["fit_seconds"]["gan_at"] = s
m, s = aug_fit("pgd_gan_at", [Xtr, Xadv_pgd_tr, Xgan_tr], [ytr, ones_aug, ones_aug])
models["pgd_gan_at"] = m
R["fit_seconds"]["pgd_gan_at"] = s
wm = np.ones(len(Xtr) + 2 * len(Xat_tr))
wm[:len(Xtr)][at_tr] = mult
wm[len(Xtr):len(Xtr) + len(Xat_tr)] = mult
m, s = aug_fit("cost_pgd_gan_at", [Xtr, Xadv_pgd_tr, Xgan_tr], [ytr, ones_aug, ones_aug], wmult=wm)
models["cost_pgd_gan_at"] = m
R["fit_seconds"]["cost_pgd_gan_at"] = s
rng_n = np.random.default_rng(seed + 3)
sub = rng_n.choice(len(Xtr), size=len(Xtr) // 2, replace=False)
Xnoisy = scaler.inverse_transform(Xtr_s[sub] + rng_n.normal(0, 0.5, Xtr_s[sub].shape))
m, s = aug_fit("noise_aug", [Xtr, Xnoisy], [ytr, ytr[sub]])
models["noise_aug"] = SmoothedModel(m, scaler, sigma=0.5, m=8, seed=seed)
R["fit_seconds"]["noise_aug"] = s
rng_n2 = np.random.default_rng(seed + 4)
sub2 = rng_n2.choice(len(Xtr), size=len(Xtr) // 2, replace=False)
Xnoisy2 = scaler.inverse_transform(Xtr_s[sub2] + rng_n2.normal(0, 0.1, Xtr_s[sub2].shape))
m, s = aug_fit("noise_aug01", [Xtr, Xnoisy2], [ytr, ytr[sub2]])
models["noise_aug01"] = SmoothedModel(m, scaler, sigma=0.1, m=8, seed=seed)
R["fit_seconds"]["noise_aug01"] = s
beh, s = cached_fit("behavioral", seed, np.delete(Xtr, IP, axis=1), ytr, w_plain, log)
R["fit_seconds"]["behavioral"] = s
save()
log("all fits done")

# ------------------------------------------------------------------ evaluation sets
Xadv_pgdA = scaler.inverse_transform(pgd_attack(sur0, Xat_s, yat, EPS, EPS / 8, 15))
ganA = AdaptiveGANAttacker(input_dim=d, noise_dim=8, random_state=seed).fit(Xb_s, Xa_s, sur0, epochs=100)
XganA = scaler.inverse_transform(ganA.generate(len(Xat)))
ganB = AdaptiveGANAttacker(input_dim=d, noise_dim=8, random_state=seed + 1000).fit(Xb_s, Xa_s, sur0, epochs=100)
XganB = scaler.inverse_transform(ganB.generate(len(Xat)))
rng = np.random.default_rng(seed + 77)
sub_bb = rng.choice(len(Xat), size=min(3000, len(Xat)), replace=False)


def cap_idx(model, n):
    if isinstance(model, SmoothedModel):
        return np.sort(rng.choice(n, size=min(args.neval_smooth, n), replace=False))
    return np.arange(n)


def eval_model(name, model, adaptive=True, blackbox=True):
    r = {}
    ic = cap_idx(model, len(Xte))
    ia = cap_idx(model, len(Xat))
    yp = model.predict(Xte[ic])
    r["clean_acc"] = float((yp == yte[ic]).mean())
    r["clean_attack_recall"] = float((yp[yte[ic] == 1] == 1).mean())
    r["clean_benign_recall"] = float((yp[yte[ic] == 0] == 0).mean())
    pc = model.predict(Xat[ia])
    ca = cost_at[ia]
    for key, Xv in (("pgdA", Xadv_pgdA), ("ganA", XganA), ("ganB", XganB)):
        pv = model.predict(Xv[ia])
        r["acc_" + key] = float((pv == 1).mean())
        if key == "pgdA":
            r["frg_pgdA_per_1000_rows"] = float(frg(pc, pv, ca) / len(ia) * 1000)
    if adaptive:
        sur_m = distill(model, scaler, Xtr, seed)
        Xp = scaler.inverse_transform(pgd_attack(sur_m, Xat_s[ia], yat[ia], EPS, EPS / 8, 15))
        r["acc_pgd_adaptive"] = float((model.predict(Xp) == 1).mean())
        gm = AdaptiveGANAttacker(input_dim=d, noise_dim=8, random_state=seed + 2000).fit(Xb_s, Xa_s, sur_m, epochs=100)
        Xg = scaler.inverse_transform(gm.generate(len(ia)))
        r["acc_gan_adaptive"] = float((model.predict(Xg) == 1).mean())
    if blackbox:
        sb = sub_bb if not isinstance(model, SmoothedModel) else sub_bb[:1500]
        adv, q = blackbox_search(model, Xat_s[sb], scaler, EPS, masks["all"], [], lo, hi, 30, np.random.default_rng(seed + 5))
        r["acc_blackbox_all"] = float((model.predict(scaler.inverse_transform(adv)) == 1).mean())
        r["blackbox_queries_per_row"] = int(q)
    r["n_attack_rows_evaluated"] = int(len(ia))
    return r


if "compare" in stages:
    R.setdefault("compare", {})
    for name in ("plain", "pgd_at", "gan_at", "pgd_gan_at", "cost_pgd_gan_at", "noise_aug", "noise_aug01"):
        if name in R["compare"]:
            continue
        t = time.time()
        R["compare"][name] = eval_model(name, models[name])
        save()
        log(f"compare/{name}: {R['compare'][name]}  ({time.time()-t:.0f}s)")

# ------------------------------------------------------------------ threat-model stress
if "threat" in stages:
    R.setdefault("threat", {})
    T = R["threat"]
    eval_names = ("plain", "pgd_at", "pgd_gan_at")
    for tm in ("all", "influenceable", "direct"):
        for eps in (0.1, 0.25, 0.5, 1.0, 2.0):
            key = f"pgd_{tm}_eps{eps}"
            if key in T:
                continue
            Xa_ = scaler.inverse_transform(masked_pgd(sur0, Xat_s, yat, eps, eps / 8, 15, masks[tm], [] if tm == 'all' else onedir, lo, hi))
            T[key] = {n: float((models[n].predict(Xa_) == 1).mean()) for n in eval_names}
            save()
            log(f"threat/{key}: {T[key]}")
    mip_ = np.zeros_like(masks["all"])
    mip_[IP] = 1.0
    mnoip_ = masks["all"].copy()
    mnoip_[IP] = 0.0
    for label_, mk_ in (("ip_only", mip_), ("no_ip", mnoip_)):
        key_ = f"pgd_{label_}_eps1.0"
        if key_ not in T:
            Xa_ = scaler.inverse_transform(masked_pgd(sur0, Xat_s, yat, 1.0, 1.0 / 8, 15, mk_, [], lo, hi))
            T[key_] = {n: float((models[n].predict(Xa_) == 1).mean()) for n in eval_names}
            save()
            log(f"threat/{key_}: {T[key_]}")
    if "indep_all" not in T:
        wq = np.random.default_rng(seed + 9).choice(len(Xtr), size=20000, replace=False)
        hard = plain.predict(Xtr[wq])
        isur = IndepSurrogate(d, seed + 21).fit_hard(Xtr_s[wq], hard)
        for tm in ("all", "influenceable"):
            Xa_ = scaler.inverse_transform(masked_pgd(isur, Xat_s, yat, EPS, EPS / 8, 15, masks[tm], [] if tm == 'all' else onedir, lo, hi))
            T[f"indep_surrogate_{tm}"] = {n: float((models[n].predict(Xa_) == 1).mean()) for n in eval_names}
        T["indep_all"] = True
        save()
        log("threat/indep done")
    if "blackbox_influenceable" not in T:
        out = {}
        for n in eval_names:
            adv, q = blackbox_search(models[n], Xat_s[sub_bb], scaler, EPS, masks["influenceable"], onedir, lo, hi, 30, np.random.default_rng(seed + 6))
            out[n] = float((models[n].predict(scaler.inverse_transform(adv)) == 1).mean())
        T["blackbox_influenceable"] = out
        save()
        log(f"threat/blackbox_influenceable {out}")
    if "blackbox_targeted" not in T:
        mip = np.zeros_like(masks["all"])
        mip[IP] = 1.0
        mnoip = masks["all"].copy()
        mnoip[IP] = 0.0
        out = {}
        for label, mk, inc in (("ip_feature_only_decrease_allowed", mip, []), ("all_but_ip_feature", mnoip, []), ("ip_feature_only_increase_only", mip, onedir)):
            out[label] = {}
            for n in eval_names:
                adv, q = blackbox_search(models[n], Xat_s[sub_bb], scaler, EPS, mk, inc, lo, hi, 30, np.random.default_rng(seed + 8))
                out[label][n] = float((models[n].predict(scaler.inverse_transform(adv)) == 1).mean())
        T["blackbox_targeted"] = out
        save()
        log(f"threat/blackbox_targeted {out}")
    if "gan_row_conditioned" not in T:
        Xg = XganA.copy()
        keep = masks["influenceable"] == 0
        Xg[:, keep] = Xat[:, keep]
        T["gan_row_conditioned"] = {n: float((models[n].predict(Xg) == 1).mean()) for n in eval_names}
        save()
        log(f"threat/gan_row_conditioned {T['gan_row_conditioned']}")

# ------------------------------------------------------------------ IP-frequency stress, cold-start mitigation
if "ipstress" in stages:
    R.setdefault("ipstress", {})
    S = R["ipstress"]

    def router_predict(Xv, thr):
        p = plain.predict(Xv)
        low = Xv[:, IP] <= thr
        if low.any():
            p[low] = beh.predict(np.delete(Xv[low], IP, axis=1))
        return p

    seq = Xat[:, IP]
    buckets = [("0", 0, 0), ("1-2", 1, 2), ("3-9", 3, 9), ("10-49", 10, 49), ("50+", 50, 10**9)]
    S["coldstart"] = {}
    p_full = plain.predict(Xat)
    p_beh = beh.predict(np.delete(Xat, IP, axis=1))
    p_r2 = router_predict(Xat, 2)
    p_r9 = router_predict(Xat, 9)
    for nm, a, b in buckets:
        mk = (seq >= a) & (seq <= b)
        S["coldstart"][nm] = {"n": int(mk.sum()),
                              "full": float((p_full[mk] == 1).mean()) if mk.any() else None,
                              "behavioral_only": float((p_beh[mk] == 1).mean()) if mk.any() else None,
                              "router_thr2": float((p_r2[mk] == 1).mean()) if mk.any() else None,
                              "router_thr9": float((p_r9[mk] == 1).mean()) if mk.any() else None}
    S["behavioral_only_overall"] = {"clean_acc": float((beh.predict(np.delete(Xte, IP, axis=1)) == yte).mean()),
                                    "attack_recall": float((p_beh == 1).mean())}
    save()
    log(f"coldstart {S['coldstart']}")
    S["router_tradeoff_clean_test"] = {}
    for nm_, fn_ in (("full_model", lambda Z: plain.predict(Z)), ("behavioral_only", lambda Z: beh.predict(np.delete(Z, IP, axis=1))),
                     ("router_thr2", lambda Z: router_predict(Z, 2)), ("router_thr9", lambda Z: router_predict(Z, 9))):
        pr_ = fn_(Xte)
        S["router_tradeoff_clean_test"][nm_] = {"accuracy": float((pr_ == yte).mean()),
                                                "attack_recall": float((pr_[yte == 1] == 1).mean()),
                                                "benign_recall": float((pr_[yte == 0] == 0).mean())}
    low_b = (Xte[:, IP] <= 9) & (yte == 0)
    S["benign_rows_with_seq_le_9_fraction"] = float(low_b.sum() / max(1, (yte == 0).sum()))
    save()
    log(f"router tradeoff {S['router_tradeoff_clean_test']}")
    rngr = np.random.default_rng(seed + 31)
    S["rotation"] = {}
    for k in (0, 2, 9, 49):
        Xr = Xat.copy()
        Xr[:, IP] = rngr.integers(0, k + 1, size=len(Xr))
        row = {n: float((models[n].predict(Xr) == 1).mean()) for n in ("plain", "pgd_at", "pgd_gan_at")}
        row["behavioral_only"] = float((beh.predict(np.delete(Xr, IP, axis=1)) == 1).mean())
        row["router_thr9"] = float((router_predict(Xr, 9) == 1).mean())
        S["rotation"][f"seq_0_to_{k}"] = row
    save()
    log(f"rotation {S['rotation']}")
    S["benign_overlap"] = {}
    Xben = Xte[~at_te]
    atk_seq = Xat_tr[:, IP]
    for f in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        Xo = Xben.copy()
        sel = rngr.random(len(Xo)) < f
        Xo[sel, IP] = rngr.choice(atk_seq, size=int(sel.sum()))
        row = {n: float((models[n].predict(Xo) == 0).mean()) for n in ("plain", "pgd_at", "pgd_gan_at")}
        row["behavioral_only"] = float((beh.predict(np.delete(Xo, IP, axis=1)) == 0).mean())
        S["benign_overlap"][f"frac_{f}"] = row
    save()
    log(f"benign_overlap {S['benign_overlap']}")

# ------------------------------------------------------------------ overhead (federated framework)
if "overhead" in stages:
    O = R.setdefault("overhead", {})
    parts = _partition_for_clients(Xtr, ytr, 3, random_state=seed)
    tenants, tsec, tmb = [], [], []
    for i, (Xc, yc) in enumerate(parts):
        stop = [False]
        peak = [rss_mb()]

        def mon():
            while not stop[0]:
                peak[0] = max(peak[0], rss_mb())
                time.sleep(0.3)

        th = threading.Thread(target=mon, daemon=True)
        th.start()
        base = rss_mb()
        mdl, sec = timed_fit(Xc, yc, balanced_sample_weight(yc), seed + i)
        stop[0] = True
        th.join()
        tenants.append(mdl)
        tsec.append(sec)
        tmb.append(peak[0] - base)
        log(f"tenant {i}: {len(Xc)} rows, {sec:.0f}s")
    O["tenant_rows"] = [len(p[0]) for p in parts]
    O["tenant_fit_seconds"] = tsec
    O["tenant_peak_rss_increase_mb"] = tmb
    O["centralized_fit_seconds"] = R["fit_seconds"]["plain"]
    O["pgd_at_fit_seconds"] = R["fit_seconds"].get("pgd_at")
    O["pickled_model_mb"] = {"centralized": len(pickle.dumps(plain)) / 2**20,
                             "tenants": [len(pickle.dumps(m_)) / 2**20 for m_ in tenants]}
    fed = FederatedEnsembleTrainer()
    fed._client_models = tenants
    rows = Xte[:20000]

    def lat(fn, Xv, reps=3):
        t0 = time.time()
        for _ in range(reps):
            fn(Xv)
        return (time.time() - t0) / reps

    def single(fn, reps=100):
        ts = []
        for i in range(reps):
            t0 = time.time()
            fn(Xte[i:i + 1])
            ts.append(time.time() - t0)
        return {"median_ms": float(np.median(ts) * 1000), "p95_ms": float(np.percentile(ts, 95) * 1000)}

    O["batch_inference_seconds_per_20000_rows"] = {"centralized": lat(plain.predict_proba, rows),
                                                   "federated_3_tenants_sequential": lat(fed.predict_proba, rows)}
    O["single_row_latency"] = {"centralized": single(plain.predict_proba), "federated_3_tenants_sequential": single(fed.predict_proba)}
    O["communication"] = {"inference_bytes_per_row_per_tenant_float64": 16,
                          "model_size_mb_per_tenant_if_models_were_shared": O["pickled_model_mb"]["tenants"],
                          "raw_data_shared_bytes": 0}
    O["attacker_side_seconds"] = {}
    t0 = time.time()
    Surrogate(input_dim=d, random_state=seed).fit(Xtr_s, plain.predict_proba(Xtr), epochs=200)
    O["attacker_side_seconds"]["surrogate_distillation"] = time.time() - t0
    t0 = time.time()
    AdaptiveGANAttacker(input_dim=d, noise_dim=8, random_state=seed).fit(Xb_s, Xa_s, sur0, epochs=100)
    O["attacker_side_seconds"]["gan_fit"] = time.time() - t0
    t0 = time.time()
    pgd_attack(sur0, Xat_s, yat, EPS, EPS / 8, 15)
    O["attacker_side_seconds"]["pgd_craft_all_attack_test_rows"] = time.time() - t0
    save()
    log(f"overhead {O}")
log("DONE")
