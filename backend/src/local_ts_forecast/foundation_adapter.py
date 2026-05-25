from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd

from .benchmark_datasets import SeriesRecord
from .io import make_future_frame, normalize_future, normalize_history, validate_prediction_length


@dataclass(frozen=True)
class FoundationAdapterTrainConfig:
    """Train only a small head on top of frozen foundation-model forecasts.

    The base Chronos/TimesFM/seasonal_naive model is never updated. During
    training, its forecasts are precomputed for rolling windows, then a small
    residual correction head is trained against the future target values.
    """

    base_backend: str = "chronos2"
    model_id: str = "autogluon/chronos-2-small"
    context_length: int = 36
    prediction_length: int = 18
    max_windows: int = 2048
    windows_per_series: int = 2
    base_batch_size: int = 32
    steps: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_size: int = 64
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 100


@dataclass(frozen=True)
class _WindowExample:
    series_id: str
    frequency: str
    context: np.ndarray
    target: np.ndarray


def _torch_device(device: str):  # noqa: ANN202
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available in this container.")
    return torch.device(device)


def _build_adapter_head(feature_size: int, hidden_size: int, dropout: float):  # noqa: ANN202
    from torch import nn

    return nn.Sequential(
        nn.Linear(feature_size, hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )


def _feature_array(
    base_forecast: np.ndarray,
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-horizon adapter features and normalized target metadata."""

    base = np.asarray(base_forecast, dtype=np.float32)
    ctx = np.asarray(context, dtype=np.float32)
    mean = float(np.mean(ctx))
    std = float(np.std(ctx))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0

    horizon = len(base)
    last = float(ctx[-1])
    first = float(ctx[0])
    trend = (last - first) / max(len(ctx) - 1, 1)
    step = np.linspace(0.0, 1.0, horizon, dtype=np.float32)

    base_norm = (base - mean) / std
    last_norm = np.full(horizon, (last - mean) / std, dtype=np.float32)
    trend_norm = np.full(horizon, trend / std, dtype=np.float32)
    step_count = np.arange(1, horizon + 1, dtype=np.float32) / max(horizon, 1)

    features = np.stack(
        [
            base_norm,
            step,
            step_count,
            last_norm,
            trend_norm,
        ],
        axis=1,
    ).astype(np.float32)
    return features, np.asarray([mean], dtype=np.float32), np.asarray([std], dtype=np.float32)


def _sample_windows(records: list[SeriesRecord], config: FoundationAdapterTrainConfig) -> list[_WindowExample]:
    rng = np.random.default_rng(config.seed)
    examples: list[_WindowExample] = []
    needed_length = config.context_length + config.prediction_length

    for record in records:
        values = np.asarray(record.values, dtype=np.float32)
        max_start = len(values) - needed_length
        if max_start < 0:
            continue

        candidates = np.arange(max_start + 1)
        rng.shuffle(candidates)
        for start in candidates[: config.windows_per_series]:
            context = values[start : start + config.context_length]
            target = values[start + config.context_length : start + needed_length]
            if len(context) == config.context_length and len(target) == config.prediction_length:
                examples.append(
                    _WindowExample(
                        series_id=record.series_id,
                        frequency=record.frequency,
                        context=context,
                        target=target,
                    )
                )
            if len(examples) >= config.max_windows:
                return examples

    rng.shuffle(examples)
    return examples


def _history_frame_for_examples(examples: list[_WindowExample]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, example in enumerate(examples):
        item_id = f"window_{idx}"
        timestamps = pd.date_range("2000-01-01", periods=len(example.context), freq=example.frequency)
        for ts, value in zip(timestamps, example.context, strict=True):
            rows.append({"id": item_id, "timestamp": ts, "target": float(value)})
    return pd.DataFrame(rows)


def _prediction_column(pred_df: pd.DataFrame) -> str:
    for column in ("predictions", "0.5", "mean"):
        if column in pred_df.columns:
            return column
    numeric_columns = [
        column
        for column in pred_df.columns
        if column not in {"id", "timestamp"} and pd.api.types.is_numeric_dtype(pred_df[column])
    ]
    if not numeric_columns:
        raise ValueError(f"No numeric prediction column found. Columns: {list(pred_df.columns)}")
    return numeric_columns[0]


def _extract_base_forecasts(pred_df: pd.DataFrame, batch_size: int, prediction_length: int) -> np.ndarray:
    pred_col = _prediction_column(pred_df)
    pred_df = pred_df.copy()
    pred_df["id"] = pred_df["id"].astype(str)

    forecasts: list[np.ndarray] = []
    for idx in range(batch_size):
        item_id = f"window_{idx}"
        group = pred_df[pred_df["id"] == item_id].sort_values("timestamp")
        values = group[pred_col].astype("float32").to_numpy()
        if len(values) < prediction_length:
            raise ValueError(
                f"Base model returned too few rows for {item_id}: "
                f"returned={len(values)}, expected={prediction_length}"
            )
        forecasts.append(values[:prediction_length])
    return np.stack(forecasts).astype(np.float32)


def _precompute_base_examples(
    examples: list[_WindowExample],
    config: FoundationAdapterTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen base model and build adapter training tensors."""

    from .forecaster import ForecastConfig, build_forecaster

    base_config = ForecastConfig(
        model_id=config.model_id,
        device=config.device,
        prediction_length=config.prediction_length,
        target="target",
        context_length=config.context_length,
        backend=config.base_backend,
        batch_size=config.base_batch_size,
    )
    base_forecaster = build_forecaster(base_config)

    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for start in range(0, len(examples), config.base_batch_size):
        batch = examples[start : start + config.base_batch_size]
        history_df = _history_frame_for_examples(batch)
        base_pred = base_forecaster.predict(history_df)
        base_forecasts = _extract_base_forecasts(base_pred, len(batch), config.prediction_length)

        for idx, example in enumerate(batch):
            features, mean, std = _feature_array(base_forecasts[idx], example.context)
            target_norm = ((example.target.astype(np.float32) - mean.item()) / std.item()).reshape(-1, 1)
            feature_batches.append(features)
            target_batches.append(target_norm.astype(np.float32))

    return np.stack(feature_batches).astype(np.float32), np.stack(target_batches).astype(np.float32)


def train_foundation_adapter(
    records: list[SeriesRecord],
    output_path: str | Path,
    config: FoundationAdapterTrainConfig,
) -> dict[str, Any]:
    """Train a residual adapter head on top of frozen Chronos/TimesFM forecasts."""

    import torch
    from torch import nn

    prediction_length = validate_prediction_length(config.prediction_length)
    if config.context_length <= 1:
        raise ValueError("context_length must be greater than 1")
    if config.max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if config.windows_per_series <= 0:
        raise ValueError("windows_per_series must be positive")

    examples = _sample_windows(records, config)
    if not examples:
        raise ValueError(
            "No trainable rolling windows were created. "
            "Increase max_series or use longer series / shorter context and prediction lengths."
        )

    started = time()
    features_np, target_np = _precompute_base_examples(examples, config)
    device = _torch_device(config.device)
    torch.manual_seed(config.seed)

    x = torch.from_numpy(features_np).to(device)
    y = torch.from_numpy(target_np).to(device)

    model = _build_adapter_head(feature_size=x.shape[-1], hidden_size=config.hidden_size, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()

    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    batch_size = min(config.batch_size, len(examples))

    model.train()
    for step in range(1, config.steps + 1):
        batch_idx = rng.integers(0, len(examples), size=batch_size)
        xb = x[batch_idx]
        yb = y[batch_idx]

        pred_residual = model(xb).squeeze(-1)
        base_norm = xb[..., 0]
        pred_norm = base_norm + pred_residual

        loss = loss_fn(pred_norm.unsqueeze(-1), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if config.log_interval and step % config.log_interval == 0:
            print(f"step={step} loss={loss_value:.6f}")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "type": "foundation_adapter",
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "feature_size": int(x.shape[-1]),
        "trained_window_count": len(examples),
        "prediction_length": prediction_length,
    }
    torch.save(checkpoint, path)

    return {
        "checkpoint": str(path),
        "base_backend": config.base_backend,
        "model_id": config.model_id,
        "trained_window_count": len(examples),
        "final_loss": losses[-1] if losses else None,
        "precompute_and_train_seconds": round(time() - started, 3),
    }


def _load_adapter_checkpoint(checkpoint_path: str | Path, device: str):  # noqa: ANN202
    import torch

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Foundation adapter checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("type") != "foundation_adapter":
        raise ValueError(f"Checkpoint is not a foundation_adapter checkpoint: {path}")

    cfg = FoundationAdapterTrainConfig(**checkpoint["config"])
    run_device = _torch_device(device if device != "auto" else cfg.device)
    model = _build_adapter_head(
        feature_size=int(checkpoint["feature_size"]),
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
    ).to(run_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, cfg, model, run_device


def _correct_predictions(
    pred_df: pd.DataFrame,
    history_df: pd.DataFrame,
    *,
    model: Any,
    device: Any,
    id_column: str,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    import torch

    corrected = pred_df.copy()
    pred_col = _prediction_column(corrected)
    corrected[id_column] = corrected[id_column].astype(str)
    history_df = history_df.copy()
    history_df[id_column] = history_df[id_column].astype(str)

    for item_id, group in corrected.groupby(id_column, sort=False):
        hist = history_df[history_df[id_column] == str(item_id)].sort_values(timestamp_column)
        if hist.empty:
            continue

        group = group.sort_values(timestamp_column)
        base_values = group[pred_col].astype("float32").to_numpy()
        context = hist[target_column].astype("float32").to_numpy()
        features, mean, std = _feature_array(base_values, context)

        with torch.no_grad():
            xb = torch.from_numpy(features).unsqueeze(0).to(device)
            residual = model(xb).squeeze(0).squeeze(-1).cpu().numpy()

        delta = residual.astype(np.float32) * std.item()
        corrected_values = base_values + delta
        idx = group.index
        corrected.loc[idx, "base_predictions"] = base_values
        corrected.loc[idx, "adapter_delta"] = delta
        corrected.loc[idx, "predictions"] = corrected_values

        for column in ("0.1", "0.5", "0.9", "mean"):
            if column in corrected.columns and pd.api.types.is_numeric_dtype(corrected[column]):
                corrected.loc[idx, column] = corrected.loc[idx, column].astype(float).to_numpy() + delta

    return corrected


def predict_with_foundation_adapter_checkpoint(
    checkpoint_path: str | Path,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame | None = None,
    *,
    target: str = "target",
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    prediction_length: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Forecast with frozen base Chronos/TimesFM plus a trained adapter head."""

    from .forecaster import ForecastConfig, build_forecaster

    _, adapter_cfg, adapter_model, run_device = _load_adapter_checkpoint(checkpoint_path, device)

    horizon = prediction_length or adapter_cfg.prediction_length
    history_df = normalize_history(history_df, target, id_column, timestamp_column)
    future_df = normalize_future(future_df, id_column, timestamp_column)
    if future_df is None:
        future_df = make_future_frame(history_df, horizon, id_column, timestamp_column)

    base_config = ForecastConfig(
        model_id=adapter_cfg.model_id,
        device=device,
        prediction_length=horizon,
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        context_length=adapter_cfg.context_length,
        backend=adapter_cfg.base_backend,
        batch_size=adapter_cfg.base_batch_size,
    )
    base_forecaster = build_forecaster(base_config)
    base_pred = base_forecaster.predict(history_df, future_df)
    return _correct_predictions(
        base_pred,
        history_df,
        model=adapter_model,
        device=run_device,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_column=target,
    )
