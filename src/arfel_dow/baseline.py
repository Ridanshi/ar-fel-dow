import numpy as np


class RateLimitBaseline:
    """Cost-threshold rate limiting: flags invocations arriving faster than
    delay_threshold as an attack. Mirrors the heuristic defenses actually
    deployed in production (per-tenant request-rate caps), used here as the
    practical-deployment comparison point against the ML detector, evaluated
    under both clean and adversarial conditions in Task 10."""

    def __init__(self, delay_threshold: float, invocation_delay_column_index: int):
        self.delay_threshold = delay_threshold
        self.invocation_delay_column_index = invocation_delay_column_index

    def predict(self, X: np.ndarray) -> np.ndarray:
        delays = X[:, self.invocation_delay_column_index]
        return (delays < self.delay_threshold).astype(np.int64)
