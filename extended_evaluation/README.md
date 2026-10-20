# Extended evaluation

Code and result files for the extended analysis of the detector in `src/arfel_dow/`.

## Contents

| Path | What it does |
|---|---|
| `rlib.py` | Shared helpers: constrained PGD, the query-only black-box search, an independent substitute model, a noise-smoothed model, invocation costs, cached model fits |
| `run_seed.py` | Per-seed runner: comparator defenses (GAN-, PGD+GAN-, cost-weighted, and noise-augmented training), defense-aware and black-box attackers, feature-subset attackers, IP-count stress tests, overhead measurements |
| `validity.py`, `sanity_ip.py`, `frg_scaling.py` | Per-seed checks of the clipped-range artifact, a no-attack count-only sanity check, and the Financial Robustness Gap per million invocations under three memory assumptions |
| `aggregate.py` | Aggregates per-seed result files into `results/aggregate.json` |
| `dow_defense.py`, `cap_sweep.py`, `synth_benign.py` | Billing-aware mitigations: evidence accumulation per source IP, and an identity-independent spend cap simulated on a synthetic bursty workload |
| `live/` | Traffic generator and evaluation for a self-hosted Apache OpenWhisk deployment; `live_s42.json`, `live_s43.json`, `live_s44.json` are the platform's own activation records for the three runs |
| `aws/` | The same traffic sent to an AWS Lambda function through the AWS SDK (`aws_gen.py`), the evaluation (`aws_eval.py`), a teardown script, and the per-invocation records (`aws_s42.json`, `aws_s43.json`, `aws_s44.json`) with the duration, billed duration, and initialization time AWS reports |
| `bccc/` | Preparation and evaluation on the BCCC-cPacket-Cloud-DDoS-2024 flow dataset |
| `bbat/` | Adversarial training against the query-only attack (rounds of retraining on evasive rows) and PGD training at several training epsilons |
| `results/` | Result files produced by the scripts above |

## Data

- **DoWTS**: public, Mendeley Data DOI 10.17632/g8g9vdxyvn.1 (see the top-level README).
- **BCCC-cPacket-Cloud-DDoS-2024**: available from the Behaviour-Centric Cybersecurity Center (York University) on request at https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/dataset-request/. It is not redistributed here. `bccc/bccc_prep.py` expects the files `Monday_18_Dec_2023.csv` and `Tuesday_19_Dec_2023.csv` from its `CSVs.zip`.
- **OpenWhisk telemetry**: included in `live/`.
- **AWS Lambda telemetry**: included in `aws/`. `aws_gen.py` expects an AWS profile named `dow-lab`; the function it calls (`dowlab-work`) is a short wait-and-return handler that you create yourself.

## Notes

- The scripts use absolute paths with the prefix `D:/Cloud RP/` (and `extended_evaluation/` under it). Replace the prefix with the location of your checkout before running.
- The OpenWhisk experiment uses the documented default test credentials of the OpenWhisk standalone distribution against a local deployment; no other credentials are involved.
- Fits of the stacked ensemble are slow on a CPU-only machine; most scripts cache fitted models and can be resumed.
