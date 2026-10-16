"""Synthetic benign workload: heavy-tailed client volume (Zipf, exponent 1), bursty arrivals
(per-bin log-normal rate multipliers), log-normal durations. All parameters are assumptions."""
import numpy as np


def benign(rng, nb, span, n_clients=40, bins=33, burst_sigma=0.8, dur_median_ms=40.0, dur_sigma=1.3, cap_ms=1000.0):
    w = 1.0 / np.arange(1, n_clients + 1)
    client = rng.choice(n_clients, size=nb, p=w / w.sum())
    mult = rng.lognormal(0.0, burst_sigma, bins)
    b = rng.choice(bins, size=nb, p=mult / mult.sum())
    t = (b + rng.random(nb)) * (span / bins)
    dur = np.minimum(rng.lognormal(np.log(dur_median_ms), dur_sigma, nb), cap_ms)
    o = np.argsort(t, kind="stable")
    return t[o], dur[o], np.array(["c%02d" % c for c in client])[o]
