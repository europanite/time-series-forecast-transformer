import numpy as np
import pandas as pd

actual = pd.read_csv("outputs/m4_monthly_holdout_actual.csv")
actual["id"] = actual["id"].astype(str)

def prepare_prediction(path: str) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred["id"] = pred["id"].astype(str)

    pred_col = "predictions" if "predictions" in pred.columns else "0.5"

    pred = pred.sort_values(["id", "timestamp"]).copy()
    pred["step"] = pred.groupby("id").cumcount() + 1

    return pred[["id", "step", pred_col]].rename(columns={pred_col: "prediction"})

def score(name: str, pred_path: str) -> None:
    pred = prepare_prediction(pred_path)

    merged = actual.merge(
        pred,
        on=["id", "step"],
        how="inner",
    )

    if merged.empty:
        raise RuntimeError(f"No matching rows for {name}. Check id and forecast step alignment.")

    y = merged["actual"].astype(float)
    yhat = merged["prediction"].astype(float)
    err = yhat - y

    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())

    nonzero = y.replace(0, np.nan).abs()
    mape = (err.abs() / nonzero).mean() * 100

    smape = (2 * err.abs() / (y.abs() + yhat.abs()).replace(0, np.nan)).mean() * 100

    print(f"[{name}]")
    print(f"rows={len(merged)}")
    print(f"MAE={mae:.6f}")
    print(f"RMSE={rmse:.6f}")
    print(f"MAPE={mape:.6f}%")
    print(f"sMAPE={smape:.6f}%")
    print()

score("zero-shot chronos2", "outputs/m4_monthly_zero_shot_chronos2.csv")
score("trained_torch", "outputs/m4_monthly_trained_torch_forecast.csv")
