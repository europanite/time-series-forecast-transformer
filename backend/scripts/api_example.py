from __future__ import annotations

import json
from pathlib import Path
from urllib import request

import pandas as pd

history = pd.read_csv("data/sample_data.csv").tail(120)
records = [
    {"id": "series_1", "timestamp": row["date"], "target": row["ITEM_A"]}
    for row in history.to_dict(orient="records")
]

payload = {
    "records": records,
    "prediction_length": 10,
    "backend": "seasonal_naive",
    "target": "target",
}

req = request.Request(
    "http://localhost:8000/forecast",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with request.urlopen(req, timeout=600) as res:
    data = json.loads(res.read().decode("utf-8"))

Path("outputs").mkdir(exist_ok=True)
Path("outputs/api_response.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("outputs/api_response.json")
