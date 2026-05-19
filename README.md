# Time-Series-Forecast-Transformer

!["web_ui"](./assets/images/web_ui.png)

Time-Series-Forecast-Transformer working at the local container.

## Run

```bash
docker compose build
docker compose up
```

Open the frontend at <http://127.0.0.1:5173>. 

The API is also available at <http://127.0.0.1:8000>.


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
