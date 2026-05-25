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
class TorchTrainConfig:
    context_length: int = 36
    prediction_length: int = 18
    steps: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 100


class TinyTimeSeriesTransformer:  # placeholder for type checkers before torch import
    pass


def _torch_device(device: str):  # noqa: ANN202
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available in this container.")
    return torch.device(device)


def _build_model(config: TorchTrainConfig):  # noqa: ANN202
    import torch
    from torch import nn

    class _TinyTimeSeriesTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_projection = nn.Linear(1, config.d_model)
            self.position = nn.Parameter(torch.zeros(1, config.context_length, config.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.prediction_length),
            )

        def forward(self, x):  # noqa: ANN001, ANN202
            x = x.unsqueeze(-1)
            h = self.value_projection(x) + self.position
            h = self.encoder(h)
            pooled = h[:, -1, :]
            return self.head(pooled)

    return _TinyTimeSeriesTransformer()


def _valid_records(records: list[SeriesRecord], total_length: int) -> list[SeriesRecord]:
    valid = [record for record in records if len(record.values) >= total_length]
    if not valid:
        raise ValueError(
            "No trainable series found. Each series must have at least "
            f"context_length + prediction_length = {total_length} observations."
        )
    return valid


def _sample_batch(records: list[SeriesRecord], config: TorchTrainConfig, rng: np.random.Generator):
    total = config.context_length + config.prediction_length
    x = np.empty((config.batch_size, config.context_length), dtype="float32")
    y = np.empty((config.batch_size, config.prediction_length), dtype="float32")

    for i in range(config.batch_size):
        record = records[int(rng.integers(0, len(records)))]
        start = int(rng.integers(0, len(record.values) - total + 1))
        window = record.values[start : start + total].astype("float32")
        context = window[: config.context_length]
        target = window[config.context_length :]
        mean = float(context.mean())
        std = float(context.std())
        if std < 1e-6:
            std = 1.0
        x[i] = (context - mean) / std
        y[i] = (target - mean) / std
    return x, y


def train_torch_forecaster(records: list[SeriesRecord], output_path: str | Path, config: TorchTrainConfig) -> dict[str, Any]:
    import torch
    from torch import nn

    if config.context_length <= 0:
        raise ValueError("context_length must be positive")
    if config.prediction_length <= 0:
        raise ValueError("prediction_length must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.steps <= 0:
        raise ValueError("steps must be positive")

    rng = np.random.default_rng(config.seed)
    valid = _valid_records(records, config.context_length + config.prediction_length)
    device = _torch_device(config.device)
    torch.manual_seed(config.seed)
    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    started = time()
    last_loss = float("nan")
    model.train()
    for step in range(1, config.steps + 1):
        x_np, y_np = _sample_batch(valid, config, rng)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if config.log_interval > 0 and (step == 1 or step % config.log_interval == 0 or step == config.steps):
            print(f"step={step} loss={last_loss:.6f}")

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "local_ts_forecast.tiny_transformer.v1",
        "model_state": model.state_dict(),
        "train_config": asdict(config),
        "trained_series_count": len(valid),
        "last_loss": last_loss,
    }
    torch.save(checkpoint, output)
    return {
        "checkpoint": str(output),
        "trained_series_count": len(valid),
        "last_loss": last_loss,
        "seconds": round(time() - started, 3),
    }


def _load_checkpoint(checkpoint_path: str | Path, device: str):
    import torch

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    map_location = _torch_device(device)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    train_config = TorchTrainConfig(**checkpoint["train_config"])
    model = _build_model(train_config).to(map_location)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, train_config, model, map_location


def predict_with_torch_checkpoint(
    checkpoint_path: str | Path,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    *,
    target: str = "target",
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    prediction_length: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    import torch

    _, train_config, model, torch_device = _load_checkpoint(checkpoint_path, device)
    horizon = validate_prediction_length(prediction_length or train_config.prediction_length)
    if horizon != train_config.prediction_length:
        raise ValueError(
            "trained_torch checkpoints currently require the same prediction_length used during training: "
            f"checkpoint={train_config.prediction_length}, requested={horizon}"
        )

    history_df = normalize_history(history_df, target, id_column, timestamp_column)
    future_df = normalize_future(future_df, id_column, timestamp_column)
    if future_df is None:
        future_df = make_future_frame(history_df, horizon, id_column, timestamp_column)

    rows: list[dict[str, object]] = []
    for item_id, group in history_df.groupby(id_column):
        group = group.sort_values(timestamp_column)
        values = group[target].astype("float32").to_numpy()
        if len(values) < train_config.context_length:
            raise ValueError(
                f"id={item_id!r} has {len(values)} observations, but checkpoint requires "
                f"context_length={train_config.context_length}."
            )
        context = values[-train_config.context_length :]
        mean = float(context.mean())
        std = float(context.std())
        if std < 1e-6:
            std = 1.0
        x = ((context - mean) / std).astype("float32")[None, :]
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(torch_device)).detach().cpu().numpy()[0]
        pred = pred * std + mean

        future_rows = future_df[future_df[id_column] == item_id].sort_values(timestamp_column).head(horizon)
        if len(future_rows) < horizon:
            raise ValueError(f"future rows for id={item_id!r} are shorter than prediction_length={horizon}")
        for step, (_, future_row) in enumerate(future_rows.iterrows()):
            y = float(pred[step])
            rows.append(
                {
                    id_column: item_id,
                    timestamp_column: future_row[timestamp_column],
                    "predictions": y,
                    "0.1": y,
                    "0.5": y,
                    "0.9": y,
                }
            )
    return pd.DataFrame(rows)
