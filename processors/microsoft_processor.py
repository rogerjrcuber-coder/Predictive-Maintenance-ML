import pandas as pd
from pathlib import Path

# ==================================================
# File Locations
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "microsoft"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_FILE = DATA_DIR / "PdM_telemetry.csv"
FAILURES_FILE = DATA_DIR / "PdM_failures.csv"
MACHINES_FILE = DATA_DIR / "PdM_machines.csv"
OUTPUT_FILE = OUTPUT_DIR / "microsoft_training.csv"


# ==================================================
# Load Datasets
# ==================================================

print("Loading datasets...")

telemetry = pd.read_csv(TELEMETRY_FILE)
failures = pd.read_csv(FAILURES_FILE)
machines = pd.read_csv(MACHINES_FILE)

print("Telemetry Shape:", telemetry.shape)
print("Failures Shape :", failures.shape)
print("Machines Shape :", machines.shape)


# ==================================================
# Convert Date Columns
# ==================================================

telemetry["datetime"] = pd.to_datetime(
    telemetry["datetime"]
)

failures["datetime"] = pd.to_datetime(
    failures["datetime"]
)

# ==================================================
# Create Failure Label
# ==================================================

telemetry["failure"] = 0

for _, failure_row in failures.iterrows():

    machine = failure_row["machineID"]
    fail_time = failure_row["datetime"]

    window_start = fail_time - pd.Timedelta(
        hours=72
    )

    mask = (
        (telemetry["machineID"] == machine)
        &
        (telemetry["datetime"] >= window_start)
        &
        (telemetry["datetime"] <= fail_time)
    )

    telemetry.loc[mask, "failure"] = 1

print(
    "Failure Records:",
    telemetry["failure"].sum()
)


# ==================================================
# Merge Machine Metadata
# ==================================================

dataset = telemetry.merge(
    machines,
    on="machineID",
    how="left"
)

# ==================================================
# Rename Columns
# ==================================================

dataset = dataset.rename(
    columns={
        "rotate": "rpm",
        "volt": "voltage"
    }
)

# ==================================================
# Keep Relevant Features
# ==================================================

training_columns = [
    "machineID",
    "datetime",
    "voltage",
    "rpm",
    "pressure",
    "vibration",
    "age",
    "failure"
]

dataset = dataset[training_columns]

# ==================================================
# Remove Missing Values
# ==================================================

dataset = dataset.dropna()

# ==================================================
# Save Processed Dataset
# ==================================================

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nProcessing Complete")
print("Saved to:", OUTPUT_FILE)
print("Final Shape:", dataset.shape)
print("\nFailure Distribution:")
print(dataset["failure"].value_counts())