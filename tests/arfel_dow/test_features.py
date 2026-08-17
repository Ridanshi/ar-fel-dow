import numpy as np
from arfel_dow.data import load_dataset
from arfel_dow.features import extract_features

FIXTURE = "data/fixtures/toy_dowts.csv"
FEATURE_COLUMNS = [
    "submit_time",
    "invocation_delay",
    "response_delay",
    "function_duration",
    "active_functions_request",
    "active_functions_response",
]


def test_extract_features_shapes_and_labels():
    df = load_dataset(FIXTURE, label_column="label")
    X, y = extract_features(df, FEATURE_COLUMNS, label_column="label", positive_label="attack")
    assert X.shape == (12, 6)
    assert y.shape == (12,)
    assert y.dtype == np.int64
    assert set(np.unique(y).tolist()) == {0, 1}
    assert y.sum() == 6  # 6 attack rows in fixture


def test_extract_features_raises_on_missing_feature_column():
    df = load_dataset(FIXTURE, label_column="label")
    try:
        extract_features(df, FEATURE_COLUMNS + ["not_a_column"], label_column="label", positive_label="attack")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not_a_column" in str(e)
