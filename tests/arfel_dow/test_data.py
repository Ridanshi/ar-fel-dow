import pandas as pd
from arfel_dow.data import load_dataset, discover_schema

FIXTURE = "data/fixtures/toy_dowts.csv"


def test_discover_schema_returns_real_headers():
    columns = discover_schema(FIXTURE)
    assert columns == [
        "submit_time",
        "invocation_delay",
        "response_delay",
        "function_duration",
        "active_functions_request",
        "active_functions_response",
        "label",
    ]


def test_load_dataset_returns_dataframe_with_label_column():
    df = load_dataset(FIXTURE, label_column="label")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert "label" in df.columns
    assert set(df["label"].unique()) == {"benign", "attack"}


def test_load_dataset_raises_if_label_column_missing():
    try:
        load_dataset(FIXTURE, label_column="not_a_real_column")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not_a_real_column" in str(e)
