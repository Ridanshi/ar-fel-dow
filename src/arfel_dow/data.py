import pandas as pd


def discover_schema(path: str) -> list[str]:
    """Return the real column headers of a CSV without loading all rows."""
    return list(pd.read_csv(path, nrows=0).columns)


def load_dataset(path: str, label_column: str) -> pd.DataFrame:
    """Load a DoWTS-format CSV, validating the label column exists."""
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(
            f"label_column '{label_column}' not found in dataset columns: {list(df.columns)}"
        )
    return df
