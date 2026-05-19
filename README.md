# Time-Series-Forecast-Transformer

Time-Series-Forecast-Transformer working at the local container.

## Run

```bash
docker compose build
docker compose up
```

Open the frontend at <http://127.0.0.1:5173>. The API is also available at <http://127.0.0.1:8000>.

Do not use the Docker bridge address printed by Vite, such as `http://172.x.x.x:5173/`, from the host browser. Use `127.0.0.1:5173`; the frontend proxies API calls through `/api` to the backend container.

The default `.env` selects `FORECAST_BACKEND=seasonal_naive` so the full-stack path can be tested without downloading a foundation model. Switch to `chronos2` or `timesfm` in the UI or `.env` when model downloads are acceptable.

## Backend CLI

Run an offline smoke forecast with the top-level sample dataset:

```bash
docker compose run --rm backend \
  python -m local_ts_forecast.cli forecast \
    --backend seasonal_naive \
    --input data/sample_data.csv \
    --target ITEM_A \
    --output outputs/forecast.csv \
    --plot outputs/forecast.png
```

The repository intentionally keeps only these sample data files:

```text
data/air_passengers.csv
data/sample_data.csv
data/sample_data.xlsx
```

Run the API example:

```bash
docker compose up -d backend
docker compose exec backend python scripts/api_example.py
```

## GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build backend
```

## Test

```bash
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```
