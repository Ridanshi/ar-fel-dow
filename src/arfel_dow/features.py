import numpy as np
import pandas as pd


def extract_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    positive_label,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns not found in dataframe: {missing}")
    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = (df[label_column] == positive_label).astype(np.int64).to_numpy()
    return X, y
