import numpy as np
from arfel_dow.baseline import RateLimitBaseline


def test_flags_high_frequency_invocations_as_attack():
    # column 1 = invocation_delay; low delay (rapid-fire calls) is the
    # classic flood signature rate-limiting is designed to catch
    X = np.array([
        [0, 12, 0, 0, 0, 0],   # normal delay -> benign
        [0, 1, 0, 0, 0, 0],    # very low delay -> flagged
        [0, 50, 0, 0, 0, 0],   # normal delay -> benign
    ])
    baseline = RateLimitBaseline(delay_threshold=5.0, invocation_delay_column_index=1)
    preds = baseline.predict(X)
    assert preds.tolist() == [0, 1, 0]
