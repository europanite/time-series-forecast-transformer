from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ACTUAL_PATH = "outputs/m4_monthly_holdout_actual.csv"
DEFAULT_PREDICTIONS = [
    "outputs/m4_monthly_zero_shot_chronos2.csv",
    "outputs/m4_monthly_trained_torch_forecast.csv",
    "outputs/m4_monthly_chronos2_adapter_forecast.csv",
]


def prepare_prediction(path: str | Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred["id"] = pred["id"].astype(str)

    pred_col = "predictions" if "predictions" in pred.columns else "0.5"
    if pred_col not in pred.columns:
        raise ValueError(f"Prediction file has no predictions/0.5 column: {path}")

    pred = pred.sort_values(["id", "timestamp"]).copy()
    pred["step"] = pred.groupby("id").cumcount() + 1
    return pred[["id", "step", pred_col]].rename(columns={pred_col: "prediction"})


def score(actual: pd.DataFrame, name: str, pred_path: str | Path) -> None:
    pred = prepare_prediction(pred_path)
    merged = actual.merge(pred, on=["id", "step"], how="inner")

    if merged.empty:
        raise RuntimeError(f"No matching rows for {name}. Check id and forecast step alignment.")

    y = merged["actual"].astype(float)
    yhat = merged["prediction"].astype(float)
    err = yhat - y

    mae = err.abs().mean()
    rmse = np.sqrt((err**2).mean())
    mape = (err.abs() / y.replace(0, np.nan).abs()).mean() * 100
    smape = (2 * err.abs() / (y.abs() + yhat.abs()).replace(0, np.nan)).mean() * 100

    print(f"[{name}]")
    print(f"path={pred_path}")
    print(f"rows={len(merged)}")
    print(f"MAE={mae:.6f}")
    print(f"RMSE={rmse:.6f}")
    print(f"MAPE={mape:.6f}%")
    print(f"sMAPE={smape:.6f}%")
    print()


def main() -> None:
    actual_path = Path(DEFAULT_ACTUAL_PATH)
    pred_paths = [Path(p) for p in (sys.argv[1:] or DEFAULT_PREDICTIONS)]

    if not actual_path.exists():
        raise FileNotFoundError(f"Actual holdout CSV not found: {actual_path}")

    actual = pd.read_csv(actual_path)
    actual["id"] = actual["id"].astype(str)

    for pred_path in pred_paths:
        if not pred_path.exists():
            print(f"skip_missing={pred_path}")
            continue
        score(actual, pred_path.stem, pred_path)


if __name__ == "__main__":
    main()
