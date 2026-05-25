from __future__ import annotations

from pathlib import Path

import pandas as pd

from local_ts_forecast.benchmark_datasets import SeriesRecord, write_records_as_long_csv
from local_ts_forecast.forecaster import ForecastConfig, build_forecaster
from local_ts_forecast.foundation_adapter import FoundationAdapterTrainConfig, train_foundation_adapter


def test_foundation_adapter_training_roundtrip(tmp_path: Path) -> None:
    records = [
        SeriesRecord(series_id="M1", values=pd.Series(range(1, 80)).to_numpy(dtype="float32"), frequency="MS"),
        SeriesRecord(series_id="M2", values=pd.Series(range(2, 81)).to_numpy(dtype="float32"), frequency="MS"),
    ]

    checkpoint = tmp_path / "foundation_adapter.pt"
    report = train_foundation_adapter(
        records,
        checkpoint,
        FoundationAdapterTrainConfig(
            base_backend="seasonal_naive",
            model_id="offline-smoke-test",
            context_length=12,
            prediction_length=3,
            max_windows=4,
            windows_per_series=2,
            base_batch_size=2,
            steps=2,
            batch_size=2,
            hidden_size=8,
            device="cpu",
            log_interval=0,
        ),
    )
    assert checkpoint.exists()
    assert report["base_backend"] == "seasonal_naive"
    assert report["trained_window_count"] == 4

    long_csv = tmp_path / "m4_long.csv"
    write_records_as_long_csv(records, long_csv)
    history_df = pd.read_csv(long_csv)

    forecaster = build_forecaster(
        ForecastConfig(
            model_id="unused",
            backend="foundation_adapter",
            checkpoint_path=str(checkpoint),
            prediction_length=3,
            target="target",
            device="cpu",
        )
    )
    pred_df = forecaster.predict(history_df)
    assert len(pred_df) == 6
    assert {"predictions", "base_predictions", "adapter_delta"}.issubset(pred_df.columns)
