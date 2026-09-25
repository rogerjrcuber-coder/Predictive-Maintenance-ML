# generate_data.py

import pandas as pd
import numpy as np

np.random.seed(42)

rows = 5000

data = pd.DataFrame({
    "temperature": np.random.normal(75, 10, rows),
    "vibration": np.random.normal(0.5, 0.2, rows),
    "pressure": np.random.normal(120, 10, rows),
    "rpm": np.random.normal(3000, 200, rows)
})

data["failure"] = (
    (data["temperature"] > 85)
    & (data["vibration"] > 0.7)
).astype(int)

data.to_csv("data/equipment_data.csv", index=False)

print("Dataset Created")