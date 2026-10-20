# AR-FEL-DoW

Adversarial-robustness evaluation of a Denial-of-Wallet (DoW) detector for serverless cloud computing.

Retargets an adversarial-resilient federated ensemble architecture (originally designed for cloud intrusion detection) at DoW attack detection, and evaluates it under gradient-based (FGSM, PGD, Carlini & Wagner) and GAN-based adaptive adversarial attacks, in addition to the clean-data accuracy figure most prior DoW detection work reports on its own.

## Authors

- Ridanshi Agarwal
- Hritwik Bali

## What's here

- `src/arfel_dow/` — the detection pipeline: stacked ensemble detector (XGBoost + RandomForest + CatBoost + MLP under a logistic-regression meta-learner), gradient and GAN-based adversarial attack suite, adversarial training defense, federated ensemble aggregation, LLM-pluggable narrative layer, and the Financial Robustness Gap metric (real AWS Lambda pricing).
- `scripts/` — experiment entry points: full evaluation run, multi-seed statistical validation, the CatBoost-substitution ablation, generalization checks (cold-start / temporal split), the federated poisoning check, the evaluation-safeguard ablation (`scripts/run_pitfall_ablation.py`: row-level versus group-by-IP split, and adversarial-training rows disjoint from versus reused for evaluation; summarize with `scripts/aggregate_pitfall_ablation.py`), and figure generation.
- `tests/` — unit tests for every module (`pytest`).
- `data/fixtures/` — a small synthetic fixture used by the test suite. The real dataset (DoWTS, 187,087 rows) is not bundled here; see below.
- `extended_evaluation/` — code and result files for the extended analysis: threat-model and defense-comparison experiments, cost analysis, billing-aware mitigations, a self-hosted OpenWhisk experiment, a second public dataset, and adversarial training against query-only attacks. See its own README.
- `results/` — result files from the multi-seed run, the CatBoost ablation, the generalization checks, the federated poisoning check, the direct train/test overfitting check, and the per-seed evaluation-safeguard ablation, kept for reproducibility.

## Dataset

Experiments run against DoWTS, a public dataset of serverless function-invocation telemetry with injected bot/DoW traffic:

> Mendeley Data, DOI [10.17632/g8g9vdxyvn.1](https://doi.org/10.17632/g8g9vdxyvn.1), CC BY 4.0.

Download it and place it at `data/raw/dowts_real.csv` before running `scripts/run_experiment.py`.

## Setup

```bash
pip install -e .
pip install -r requirements.txt
pytest
```

## Running the full evaluation

```bash
python scripts/run_experiment.py
```

CPU-only; no GPU required. The full run (187,087 rows, 5-seed statistical validation via `scripts/run_multi_seed.py`) takes on the order of hours on a single machine, since the stacked ensemble is refit from scratch per seed.

## Key findings

- Clean accuracy: 99.42% ± 0.05% (mean ± std, 5 seeds), competitive with prior clean-data-only DoW detectors.
- Accuracy under PGD evasion, undefended: 63.94% ± 1.96%. Adversarially trained: 100.00% ± 0.00%.
- Accuracy under a GAN-based adaptive attack, undefended: 47.66% ± 4.51%. Adversarially trained (PGD-only): 86.65% ± 2.18%, demonstrating partial cross-attack-method generalization.
- A production-style rate-limiting baseline catches close to none of this traffic, clean or adversarial.
- The Financial Robustness Gap metric quantifies the dollar cost of the clean-vs-adversarial detection gap using real AWS Lambda pricing.

## License

MIT. See `LICENSE`.
