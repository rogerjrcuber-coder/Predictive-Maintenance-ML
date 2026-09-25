# inspect_microsoft.py

import pandas as pd

telemetry = pd.read_csv(
    "data/microsoft/PdM_telemetry.csv"
)

print(telemetry.head())
print("\nColumns:")
print(telemetry.columns)
print("\nShape:")
print(telemetry.shape)