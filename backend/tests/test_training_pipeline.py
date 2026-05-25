from __future__ import annotations

from pathlib import Path

import pandas as pd

from local_ts_forecast.benchmark_datasets import load_m4_series, load_m5_series, write_records_as_long_csv
from local_ts_forecast.forecaster import ForecastConfig, build_forecaster
from local_ts_forecast.torch_training import TorchTrainConfig, train_torch_forecaster


def test_m4_loader_and_torch_training_roundtrip(tmp_path: Path) -> None:
    train_dir = tmp_path / "m4" / "Train"
    train_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            ["M1", *range(1, 80)],
            ["M2", *range(2, 81)],
        ]
    ).to_csv(train_dir / "Monthly-train.csv", index=False)

    records = load_m4_series(train_dir, "Monthly", min_length=30)
    assert len(records) == 2

    checkpoint = tmp_path / "model.pt"
    report = train_torch_forecaster(
        records,
        checkpoint,
        TorchTrainConfig(
            context_length=12,
            prediction_length=3,
            steps=2,
            batch_size=2,
            d_model=16,
            num_layers=1,
            num_heads=2,
            device="cpu",
            log_interval=0,
        ),
    )
    assert checkpoint.exists()
    assert report["trained_series_count"] == 2

    long_csv = tmp_path / "m4_long.csv"
    write_records_as_long_csv(records, long_csv)
    history_df = pd.read_csv(long_csv)
    forecaster = build_forecaster(
        ForecastConfig(
            model_id="offline-trained-torch",
            backend="trained_torch",
            checkpoint_path=str(checkpoint),
            prediction_length=3,
            context_length=12,
            target="target",
            device="cpu",
        )
    )
    pred_df = forecaster.predict(history_df)
    assert len(pred_df) == 6
    assert {"id", "timestamp", "predictions"}.issubset(pred_df.columns)


def test_m5_loader(tmp_path: Path) -> None:
    m5_dir = tmp_path / "m5"
    m5_dir.mkdir()
    pd.DataFrame(
        {
            "id": ["item_1_store_1_validation", "item_2_store_1_validation"],
            "item_id": ["item_1", "item_2"],
            "dept_id": ["dept_1", "dept_1"],
            "cat_id": ["cat_1", "cat_1"],
            "store_id": ["store_1", "store_1"],
            "state_id": ["CA", "CA"],
            "d_1": [1, 4],
            "d_2": [2, 5],
            "d_3": [3, 6],
        }
    ).to_csv(m5_dir / "sales_train_validation.csv", index=False)

    records = load_m5_series(m5_dir, min_length=3)
    assert len(records) == 2
    assert records[0].frequency == "D"
    assert records[0].values.tolist() == [1.0, 2.0, 3.0]
