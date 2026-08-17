import numpy as np
from arfel_dow.pricing import AWSLambdaPricing, financial_robustness_gap


def test_invocation_cost_matches_published_formula():
    # AWS Lambda on-demand pricing (x86, us-east-1), as published by AWS and
    # cited in this project's design doc — accessed 2026-08-12. Re-verify
    # against https://aws.amazon.com/lambda/pricing/ before quoting in the
    # paper, since cloud pricing changes over time.
    pricing = AWSLambdaPricing()
    # 1 invocation, 1000ms duration, 128MB memory:
    # request cost = 0.20 / 1_000_000 = 0.0000002
    # compute cost = 1000ms/1000 * (128/1024)GB * 0.0000166667 $/GB-s
    #              = 1.0 * 0.125 * 0.0000166667 = 0.0000020833...
    # total = 0.0000002 + 0.0000020833 = 0.0000022833...
    cost = pricing.invocation_cost(duration_ms=1000, memory_mb=128, n_invocations=1)
    assert abs(cost - 0.0000022833) < 1e-9


def test_invocation_cost_scales_linearly_with_invocation_count():
    pricing = AWSLambdaPricing()
    cost_1 = pricing.invocation_cost(duration_ms=500, memory_mb=256, n_invocations=1)
    cost_100 = pricing.invocation_cost(duration_ms=500, memory_mb=256, n_invocations=100)
    assert abs(cost_100 - cost_1 * 100) < 1e-9


def test_financial_robustness_gap_is_zero_when_predictions_identical():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    durations = np.array([1000, 1000, 1000, 1000])
    memory = np.array([128, 128, 128, 128])
    gap = financial_robustness_gap(y_true, y_pred, y_pred, durations, memory)
    assert gap == 0.0


def test_financial_robustness_gap_positive_when_adversarial_misses_more_attacks():
    y_true = np.array([1, 1, 1, 1])
    y_pred_clean = np.array([1, 1, 1, 0])       # 1 missed attack
    y_pred_adversarial = np.array([0, 0, 1, 0])  # 2 missed attacks
    durations = np.array([5000, 5000, 5000, 5000])
    memory = np.array([1024, 1024, 1024, 1024])
    gap = financial_robustness_gap(y_true, y_pred_clean, y_pred_adversarial, durations, memory)
    assert gap > 0.0
