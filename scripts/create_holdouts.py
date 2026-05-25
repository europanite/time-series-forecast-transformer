from pathlib import Path

import pandas as pd

freq = "Monthly"
horizon = 18
max_series = 10000

train_path = Path("data/m4/Train") / f"{freq}-train.csv"
test_path = Path("data/m4/Test") / f"{freq}-test.csv"
output_path = Path("outputs/m4_monthly_holdout_actual.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

id_col = train.columns[0]
value_cols = list(test.columns[1:])

# Use the same first N series as the converted training data.
train = train.head(max_series)
test = test[test[id_col].isin(train[id_col].astype(str))].head(max_series)

rows = []
for _, row in test.iterrows():
    series_id = str(row[id_col])
    for step, col in enumerate(value_cols[:horizon], start=1):
        value = row[col]
        if pd.isna(value):
            continue
        rows.append(
            {
                "id": series_id,
                "step": step,
                "actual": float(value),
            }
        )

actual = pd.DataFrame(rows)
actual.to_csv(output_path, index=False)
print(f"actual_csv={output_path}")
print(f"rows={len(actual)}")