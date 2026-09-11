"""Defensive dataframe cleaning and exploratory summaries."""

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def impute_missing_values(frame: pd.DataFrame, numeric_strategy: str = "median") -> pd.DataFrame:
    """Impute numeric columns and preserve non-numeric columns."""
    if frame.empty:
        return frame.copy()
    if numeric_strategy not in {"mean", "median", "most_frequent"}:
        raise ValueError("numeric_strategy must be mean, median, or most_frequent")
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        series = result[column]
        if not series.isna().any():
            continue
        if numeric_strategy == "mean":
            value = series.mean()
        elif numeric_strategy == "median":
            value = series.median()
        else:
            modes = series.mode(dropna=True)
            value = modes.iloc[0] if not modes.empty else 0.0
        result.loc[:, column] = series.fillna(value)
    for column in result.select_dtypes(exclude=[np.number]).columns:
        if result[column].isna().any():
            result.loc[:, column] = result[column].fillna("unknown")
    return result


def cap_iqr_outliers(frame: pd.DataFrame, columns: Iterable[str] | None = None, multiplier: float = 1.5) -> pd.DataFrame:
    """Winsorize numeric outliers to Tukey IQR fences."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    result = frame.copy()
    selected = list(columns) if columns is not None else list(result.select_dtypes(include=[np.number]).columns)
    missing = set(selected) - set(result.columns)
    if missing:
        raise KeyError(f"unknown columns: {sorted(missing)}")
    for column in selected:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"column is not numeric: {column}")
        q1, q3 = result[column].quantile([0.25, 0.75])
        spread = q3 - q1
        result.loc[:, column] = result[column].clip(q1 - multiplier * spread, q3 + multiplier * spread)
    return result


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric fields."""
    if frame.empty:
        raise ValueError("frame must not be empty")
    return frame.select_dtypes(include=[np.number]).describe().T
