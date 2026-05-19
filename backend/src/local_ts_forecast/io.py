from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_HISTORY_COLUMNS = {"id", "timestamp"}
REQUIRED_FUTURE_COLUMNS = {"id", "timestamp"}
DATETIME_CANDIDATES = ("timestamp", "datetime", "date", "time", "ds", "Date")


class DataValidationError(ValueError):
    pass


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError(f"CSV is empty: {path}")
    return df


def _target_columns(target: str | Iterable[str]) -> list[str]:
    if isinstance(target, str):
        return [target]
    values = list(target)
    if not values:
        raise DataValidationError("target columns must not be empty")
    return values


def _ensure_timestamp_column(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    if timestamp_column in df.columns:
        return df
    for candidate in DATETIME_CANDIDATES:
        if candidate in df.columns:
            df = df.copy()
            df[timestamp_column] = df[candidate]
            return df
    return df


def normalize_history(
    df: pd.DataFrame,
    target: str | Iterable[str] = "target",
    id_column: str = "id",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    df = _ensure_timestamp_column(df.copy(), timestamp_column)
    if id_column not in df.columns:
        df.insert(0, id_column, "series_1")

    target_columns = _target_columns(target)
    missing = (REQUIRED_HISTORY_COLUMNS - set(df.columns)) | (set(target_columns) - set(df.columns))
    if missing:
        raise DataValidationError(
            "history CSV must contain id/timestamp and the selected target column. "
            f"Missing: {sorted(missing)}"
        )

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")
    for column in target_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df = df.sort_values([id_column, timestamp_column]).reset_index(drop=True)
    return df


def normalize_future(
    df: pd.DataFrame | None,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame | None:
    if df is None:
        return None
    df = _ensure_timestamp_column(df.copy(), timestamp_column)
    if id_column not in df.columns:
        df.insert(0, id_column, "series_1")

    missing = REQUIRED_FUTURE_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            "future CSV must contain id/timestamp. "
            f"Missing: {sorted(missing)}"
        )

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")
    df = df.sort_values([id_column, timestamp_column]).reset_index(drop=True)
    return df


def validate_prediction_length(prediction_length: int) -> int:
    if prediction_length <= 0:
        raise DataValidationError("prediction_length must be positive")
    return prediction_length


def infer_frequency(df: pd.DataFrame, id_column: str = "id", timestamp_column: str = "timestamp") -> str | None:
    freqs: list[str] = []
    for _, group in df.groupby(id_column):
        freq = pd.infer_freq(group[timestamp_column])
        if freq:
            freqs.append(freq)
    if not freqs:
        return None
    return freqs[0]


def make_future_frame(
    history_df: pd.DataFrame,
    prediction_length: int,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Create future id/timestamp rows when no known covariate file is supplied."""
    rows: list[dict[str, object]] = []
    for item_id, group in history_df.groupby(id_column):
        group = group.sort_values(timestamp_column)
        freq = pd.infer_freq(group[timestamp_column])
        if freq is None:
            raise DataValidationError(
                f"Could not infer timestamp frequency for id={item_id!r}. "
                "Provide future-input CSV with explicit timestamps."
            )
        last_ts = group[timestamp_column].iloc[-1]
        future_index = pd.date_range(start=last_ts, periods=prediction_length + 1, freq=freq)[1:]
        for ts in future_index:
            rows.append({id_column: item_id, timestamp_column: ts})
    return pd.DataFrame(rows)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def select_target_columns(target: str | Iterable[str]) -> str | list[str]:
    if isinstance(target, str):
        return target
    values = list(target)
    if not values:
        raise DataValidationError("target columns must not be empty")
    return values