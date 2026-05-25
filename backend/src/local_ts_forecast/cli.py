from __future__ import annotations

import argparse
import os
from pathlib import Path

from .benchmark_datasets import load_training_series, write_records_as_long_csv
from .forecaster import ForecastConfig, build_forecaster
from .foundation_adapter import FoundationAdapterTrainConfig, train_foundation_adapter
from .io import ensure_parent, read_csv
from .plotting import plot_forecast
from .sample_data import write_sample_data
from .settings import get_settings
from .torch_training import TorchTrainConfig, train_torch_forecaster


M4_FREQUENCY_CHOICES = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]


def parse_quantiles(value: str) -> tuple[float, ...]:
    values = tuple(float(v.strip()) for v in value.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("quantiles must not be empty")
    for v in values:
        if not 0 < v < 1:
            raise argparse.ArgumentTypeError("quantiles must be between 0 and 1")
    return values


def forecast_command(args: argparse.Namespace) -> None:
    settings = get_settings()
    history_df = read_csv(args.input)
    future_df = read_csv(args.future_input) if args.future_input else None

    config = ForecastConfig(
        model_id=args.model_id or settings.model_id_for_backend(args.backend),
        device=args.device or settings.device,
        prediction_length=args.prediction_length or settings.prediction_length,
        quantile_levels=args.quantiles,
        target=args.target,
        batch_size=args.batch_size,
        context_length=args.context_length,
        backend=args.backend,
        checkpoint_path=args.checkpoint,
    )
    forecaster = build_forecaster(config)
    pred_df = forecaster.predict(history_df, future_df)

    output_path = ensure_parent(args.output)
    pred_df.to_csv(output_path, index=False)
    print(f"forecast_csv={output_path}")

    if args.plot:
        plot_path = plot_forecast(history_df, pred_df, args.plot, target_column=args.target)
        print(f"forecast_plot={plot_path}")


def train_command(args: argparse.Namespace) -> None:
    min_length = args.min_length or (args.context_length + args.prediction_length)
    records = load_training_series(
        args.dataset,
        m4_train_dir=args.m4_train_dir,
        m4_frequency=args.m4_frequency,
        m5_dir=args.m5_dir,
        csv_path=args.input,
        target=args.target,
        max_series=args.max_series,
        min_length=min_length,
    )
    if not records:
        raise SystemExit("No trainable series were loaded.")

    config = TorchTrainConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    report = train_torch_forecaster(records, args.output, config)
    print(f"dataset={args.dataset}")
    print(f"loaded_series={len(records)}")
    for key, value in report.items():
        print(f"{key}={value}")


def train_adapter_command(args: argparse.Namespace) -> None:
    min_length = args.min_length or (args.context_length + args.prediction_length)
    settings = get_settings()
    base_backend = args.base_backend
    records = load_training_series(
        args.dataset,
        m4_train_dir=args.m4_train_dir,
        m4_frequency=args.m4_frequency,
        m5_dir=args.m5_dir,
        csv_path=args.input,
        target=args.target,
        max_series=args.max_series,
        min_length=min_length,
    )
    if not records:
        raise SystemExit("No trainable series were loaded.")

    config = FoundationAdapterTrainConfig(
        base_backend=base_backend,
        model_id=args.model_id or settings.model_id_for_backend(base_backend),
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        max_windows=args.max_windows,
        windows_per_series=args.windows_per_series,
        base_batch_size=args.base_batch_size,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    report = train_foundation_adapter(records, args.output, config)
    print(f"dataset={args.dataset}")
    print(f"loaded_series={len(records)}")
    for key, value in report.items():
        print(f"{key}={value}")


def convert_dataset_command(args: argparse.Namespace) -> None:
    min_length = args.min_length or 1
    records = load_training_series(
        args.dataset,
        m4_train_dir=args.m4_train_dir,
        m4_frequency=args.m4_frequency,
        m5_dir=args.m5_dir,
        csv_path=args.input,
        target=args.target,
        max_series=args.max_series,
        min_length=min_length,
    )
    if not records:
        raise SystemExit("No series were loaded.")
    output_path = write_records_as_long_csv(records, args.output, max_rows=args.max_rows)
    print(f"dataset={args.dataset}")
    print(f"series={len(records)}")
    print(f"output_csv={output_path}")


def sample_data_command(args: argparse.Namespace) -> None:
    history_path, future_path = write_sample_data(
        output_dir=args.output_dir,
        periods=args.periods,
        prediction_length=args.prediction_length,
        freq=args.freq,
    )
    print(f"history_csv={history_path}")
    print(f"future_csv={future_path}")


def validate_command(args: argparse.Namespace) -> None:
    history_df = read_csv(args.input)
    future_df = read_csv(args.future_input) if args.future_input else None
    config = ForecastConfig(
        model_id="offline-smoke-test",
        device="cpu",
        prediction_length=args.prediction_length,
        backend="seasonal_naive",
    )
    pred_df = build_forecaster(config).predict(history_df, future_df)
    print("validation=ok")
    print(f"rows={len(pred_df)}")


def add_training_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["m4", "m5", "csv"], default="m4")
    parser.add_argument("--m4-train-dir", default="data_private/m4/Train", help="Directory containing M4 *-train.csv files")
    parser.add_argument("--m4-frequency", choices=M4_FREQUENCY_CHOICES, default="Monthly")
    parser.add_argument("--m5-dir", default="data_private/m5", help="Directory containing M5 Walmart CSV files")
    parser.add_argument("--input", help="Long CSV path when --dataset csv")
    parser.add_argument("--target", default="target")
    parser.add_argument("--max-series", type=int, help="Limit series count for quick experiments")
    parser.add_argument("--min-length", type=int, help="Minimum observations per series")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local time-series forecasting with Chronos-2, TimesFM, or local Torch training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample-data", help="Generate synthetic electricity-like sample data")
    sample.add_argument("--output-dir", default="data", help="Directory to write sample CSVs")
    sample.add_argument("--periods", type=int, default=14 * 48, help="History length")
    sample.add_argument("--prediction-length", type=int, default=int(os.getenv("PREDICTION_LENGTH", "48")))
    sample.add_argument("--freq", default="30min")
    sample.set_defaults(func=sample_data_command)

    forecast = subparsers.add_parser("forecast", help="Run forecasting")
    forecast.add_argument("--input", required=True, help="History CSV. Required columns: timestamp,target. Optional: id")
    forecast.add_argument("--future-input", help="Future covariate CSV. Required columns: timestamp. Optional: id/covariates")
    forecast.add_argument("--output", default="outputs/forecast.csv")
    forecast.add_argument("--plot", default="outputs/forecast.png")
    forecast.add_argument("--prediction-length", type=int, default=int(os.getenv("PREDICTION_LENGTH", "48")))
    forecast.add_argument("--model-id", help="Hugging Face model id. Defaults to CHRONOS_MODEL_ID or TIMESFM_MODEL_ID based on backend.")
    forecast.add_argument("--device", default=os.getenv("DEVICE", "cpu"), choices=["cpu", "cuda", "auto"])
    forecast.add_argument("--quantiles", type=parse_quantiles, default=(0.1, 0.5, 0.9))
    forecast.add_argument("--target", default="target")
    forecast.add_argument("--batch-size", type=int)
    forecast.add_argument("--context-length", type=int)
    forecast.add_argument("--checkpoint", help="Checkpoint path for --backend trained_torch or foundation_adapter")
    forecast.add_argument(
        "--backend",
        choices=["chronos2", "timesfm", "seasonal_naive", "trained_torch", "foundation_adapter"],
        default=os.getenv("FORECAST_BACKEND", "chronos2"),
    )
    forecast.set_defaults(func=forecast_command)

    train = subparsers.add_parser("train", help="Train a local Torch global forecaster from M4, M5, or long CSV data")
    add_training_dataset_args(train)
    train.add_argument("--output", default="outputs/trained_torch.pt")
    train.add_argument("--context-length", type=int, default=36)
    train.add_argument("--prediction-length", type=int, default=18)
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--d-model", type=int, default=64)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--num-heads", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--device", default=os.getenv("DEVICE", "cpu"), choices=["cpu", "cuda", "auto"])
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log-interval", type=int, default=100)
    train.set_defaults(func=train_command)

    adapter = subparsers.add_parser(
        "train-adapter",
        help="Freeze Chronos/TimesFM/seasonal_naive and train only a residual adapter head on M4, M5, or long CSV data",
    )
    add_training_dataset_args(adapter)
    adapter.add_argument("--output", default="outputs/foundation_adapter.pt")
    adapter.add_argument("--base-backend", choices=["chronos2", "timesfm", "seasonal_naive"], default="chronos2")
    adapter.add_argument("--model-id", help="Base model id. Defaults to CHRONOS_MODEL_ID or TIMESFM_MODEL_ID based on --base-backend.")
    adapter.add_argument("--context-length", type=int, default=36)
    adapter.add_argument("--prediction-length", type=int, default=18)
    adapter.add_argument("--max-windows", type=int, default=2048)
    adapter.add_argument("--windows-per-series", type=int, default=2)
    adapter.add_argument("--base-batch-size", type=int, default=32)
    adapter.add_argument("--steps", type=int, default=1000)
    adapter.add_argument("--batch-size", type=int, default=128)
    adapter.add_argument("--learning-rate", type=float, default=1e-3)
    adapter.add_argument("--hidden-size", type=int, default=64)
    adapter.add_argument("--dropout", type=float, default=0.1)
    adapter.add_argument("--device", default=os.getenv("DEVICE", "cpu"), choices=["cpu", "cuda", "auto"])
    adapter.add_argument("--seed", type=int, default=42)
    adapter.add_argument("--log-interval", type=int, default=100)
    adapter.set_defaults(func=train_adapter_command)

    convert = subparsers.add_parser("convert-dataset", help="Convert M4/M5/long CSV data to id,timestamp,target CSV")
    add_training_dataset_args(convert)
    convert.add_argument("--output", required=True)
    convert.add_argument("--max-rows", type=int)
    convert.set_defaults(func=convert_dataset_command)

    validate = subparsers.add_parser("validate", help="Validate CSV and pipeline without downloading a model")
    validate.add_argument("--input", required=True)
    validate.add_argument("--future-input")
    validate.add_argument("--prediction-length", type=int, default=int(os.getenv("PREDICTION_LENGTH", "48")))
    validate.set_defaults(func=validate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
