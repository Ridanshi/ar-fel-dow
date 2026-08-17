import numpy as np


class AWSLambdaPricing:
    """AWS Lambda on-demand pricing (x86, us-east-1), as published by AWS.
    Source: https://aws.amazon.com/lambda/pricing/ — accessed 2026-08-12.
    These constants must be re-verified against the live pricing page before
    any dollar figure from this module is quoted in the paper; cloud pricing
    changes over time and this class has no auto-update mechanism."""

    REQUEST_COST_PER_INVOCATION = 0.20 / 1_000_000  # $0.20 per 1M requests
    COMPUTE_COST_PER_GB_SECOND = 0.0000166667        # $ per GB-second

    def invocation_cost(self, duration_ms: float, memory_mb: float, n_invocations: int) -> float:
        duration_s = duration_ms / 1000.0
        memory_gb = memory_mb / 1024.0
        compute_cost = duration_s * memory_gb * self.COMPUTE_COST_PER_GB_SECOND
        request_cost = self.REQUEST_COST_PER_INVOCATION
        return (compute_cost + request_cost) * n_invocations


def financial_robustness_gap(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adversarial: np.ndarray,
    durations_ms: np.ndarray,
    memory_mb: np.ndarray,
    pricing: AWSLambdaPricing | None = None,
) -> float:
    """Dollar cost of missed-attack invocations under adversarial evasion,
    minus the dollar cost of missed-attack invocations on clean data. A
    missed attack (false negative) is billed at its real invocation cost
    since the attacker's traffic continues uninterrupted."""
    if pricing is None:
        pricing = AWSLambdaPricing()

    missed_clean = (y_true == 1) & (y_pred_clean == 0)
    missed_adversarial = (y_true == 1) & (y_pred_adversarial == 0)

    cost_clean = sum(
        pricing.invocation_cost(durations_ms[i], memory_mb[i], n_invocations=1)
        for i in np.where(missed_clean)[0]
    )
    cost_adversarial = sum(
        pricing.invocation_cost(durations_ms[i], memory_mb[i], n_invocations=1)
        for i in np.where(missed_adversarial)[0]
    )

    return cost_adversarial - cost_clean
