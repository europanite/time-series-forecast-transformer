from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesRecord:
    series_id: str
    values: np.ndarray
    frequency: str


M4_FREQUENCIES = {
    "yearly": ("Yearly", "YS"),
    "quarterly": ("Quarterly", "QS"),
    "monthly": ("Monthly", "MS"),
    "weekly": ("Weekly", "W"),
    "daily": ("Daily", "D"),
    "hourly": ("Hourly", "h"),
}


def _expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _canonical_m4_frequency(frequency: str) -> tuple[str, str]:
    key = frequency.strip().lower()
    if key not in M4_FREQUENCIES:
        allowed = ", ".join(v[0] for v in M4_FREQUENCIES.values())
        raise ValueError(f"Unknown M4 frequency: {frequency!r}. Allowed: {allowed}")
    return M4_FREQUENCIES[key]


def _clean_values(values: Iterable[object]) -> np.ndarray:
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype("float32")
    return numeric.to_numpy()


def load_m4_series(
    train_dir: str | Path,
    frequency: str = "Monthly",
    max_series: int | None = None,
    min_length: int = 1,
) -> list[SeriesRecord]:
    """Load M4 train CSV files into in-memory numeric series.

    M4 files are wide CSVs such as `Monthly-train.csv`: the first column is the
    series id, and the remaining columns are observations. M4 does not provide
    real timestamps, so downstream conversion creates synthetic timestamps using
    the selected frequency.
    """

    canonical, pandas_freq = _canonical_m4_frequency(frequency)
    path = _expand_path(train_dir) / f"{canonical}-train.csv"
    if not path.exists():
        raise FileNotFoundError(f"M4 train file not found: {path}")

    df = pd.read_csv(path)
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"M4 train file has no usable series columns: {path}")

    records: list[SeriesRecord] = []
    for _, row in df.iterrows():
        series_id = str(row.iloc[0])
        values = _clean_values(row.iloc[1:])
        if len(values) >= min_length:
            records.append(SeriesRecord(series_id=series_id, values=values, frequency=pandas_freq))
        if max_series is not None and len(records) >= max_series:
            break
    return records


def _find_m5_sales_file(input_dir: Path) -> Path:
    candidates = (
        input_dir / "sales_train_evaluation.csv",
        input_dir / "sales_train_validation.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "M5 sales file not found. Expected sales_train_evaluation.csv or "
        f"sales_train_validation.csv under {input_dir}"
    )


def load_m5_series(
    input_dir: str | Path,
    max_series: int | None = None,
    min_length: int = 1,
) -> list[SeriesRecord]:
    """Load M5 Walmart sales data into in-memory daily series."""

    root = _expand_path(input_dir)
    sales_path = _find_m5_sales_file(root)
    df = pd.read_csv(sales_path)
    d_columns = [col for col in df.columns if col.startswith("d_")]
    if not d_columns:
        raise ValueError(f"M5 sales file has no d_* columns: {sales_path}")

    records: list[SeriesRecord] = []
    for _, row in df.iterrows():
        series_id = str(row["id"]) if "id" in row.index else str(row.iloc[0])
        values = _clean_values(row[d_columns])
        if len(values) >= min_length:
            records.append(SeriesRecord(series_id=series_id, values=values, frequency="D"))
        if max_series is not None and len(records) >= max_series:
            break
    return records


def load_long_csv_series(
    path: str | Path,
    target: str = "target",
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    max_series: int | None = None,
    min_length: int = 1,
) -> list[SeriesRecord]:
    csv_path = _expand_path(path)
    df = pd.read_csv(csv_path)
    if id_column not in df.columns:
        df.insert(0, id_column, "series_1")
    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")
        df = df.sort_values([id_column, timestamp_column])
    if target not in df.columns:
        raise ValueError(f"target column not found in {csv_path}: {target}")

    records: list[SeriesRecord] = []
    for series_id, group in df.groupby(id_column):
        values = _clean_values(group[target])
        if len(values) >= min_length:
            records.append(SeriesRecord(series_id=str(series_id), values=values, frequency="D"))
        if max_series is not None and len(records) >= max_series:
            break
    return records


def load_training_series(
    dataset: str,
    *,
    m4_train_dir: str | Path = "data_private/m4/Train",
    m4_frequency: str = "Monthly",
    m5_dir: str | Path = "data_private/m5",
    csv_path: str | Path | None = None,
    target: str = "target",
    max_series: int | None = None,
    min_length: int = 1,
) -> list[SeriesRecord]:
    selected = dataset.lower()
    if selected == "m4":
        return load_m4_series(m4_train_dir, m4_frequency, max_series=max_series, min_length=min_length)
    if selected == "m5":
        return load_m5_series(m5_dir, max_series=max_series, min_length=min_length)
    if selected == "csv":
        if csv_path is None:
            raise ValueError("--input is required when --dataset csv")
        return load_long_csv_series(csv_path, target=target, max_series=max_series, min_length=min_length)
    raise ValueError(f"Unknown training dataset: {dataset}")


def records_to_long_frame(records: list[SeriesRecord], max_rows: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        timestamps = pd.date_range("2000-01-01", periods=len(record.values), freq=record.frequency)
        for ts, value in zip(timestamps, record.values, strict=True):
            rows.append({"id": record.series_id, "timestamp": ts, "target": float(value)})
            if max_rows is not None and len(rows) >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def write_records_as_long_csv(records: list[SeriesRecord], output_path: str | Path, max_rows: int | None = None) -> Path:
    path = _expand_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_long_frame(records, max_rows=max_rows)
    df.to_csv(path, index=False)
    return path
