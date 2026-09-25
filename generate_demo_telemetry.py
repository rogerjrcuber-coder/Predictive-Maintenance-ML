import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "synthetic" / "demo"
ROWS = 5000
INTERVAL = timedelta(minutes=5)
START_TIME = datetime(2026, 1, 1)
RANDOM_SEED = 42


PROFILES = {
    "healthy": {
        "machine_id": "PUMP-HEALTHY",
        "voltage": (475, 485),
        "rpm": (2950, 3050),
        "pressure": (98, 105),
        "vibration": (1, 3),
        "temperature": (60, 70),
        "load": (35, 60),
        "degradation": 0.0,
        "failure_threshold": 999,
    },
    "monitor": {
        "machine_id": "PUMP-MONITOR",
        "voltage": (470, 490),
        "rpm": (2900, 3200),
        "pressure": (90, 115),
        "vibration": (3, 6),
        "temperature": (70, 85),
        "load": (55, 80),
        "degradation": 0.35,
        "failure_threshold": 0.92,
    },
    "critical": {
        "machine_id": "PUMP-CRITICAL",
        "voltage": (455, 485),
        "rpm": (3100, 3600),
        "pressure": (115, 145),
        "vibration": (8, 15),
        "temperature": (90, 120),
        "load": (75, 98),
        "degradation": 0.85,
        "failure_threshold": 0.65,
    },
}


CSV_COLUMNS = [
    "machine_id",
    "timestamp",
    "voltage",
    "rpm",
    "pressure_psi",
    "vibration_mm_s",
    "temperature_c",
    "current",
    "power_kw",
    "load_pct",
    "operating_state",
    "maintenance_count",
    "error_count_24h",
    "error_count_7d",
    "age",
    "vibration_delta",
    "pressure_delta",
    "temperature_delta",
    "failure",
]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def generate_machine(profile_name, profile):
    output_path = OUTPUT_DIR / f"{profile_name}_machine.csv"
    age = random.randint(2, 15)
    maintenance_count = random.randint(0, 8)
    timestamp = START_TIME
    previous_vibration = None
    previous_pressure = None
    previous_temperature = None

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row_number in range(ROWS):
            progress = row_number / (ROWS - 1)
            degradation = profile["degradation"] * progress
            voltage = random.uniform(*profile["voltage"]) - degradation * 8
            rpm = random.uniform(*profile["rpm"]) + degradation * 200
            pressure = random.uniform(*profile["pressure"]) + degradation * 8
            vibration = random.uniform(*profile["vibration"]) + degradation * 3
            temperature = random.uniform(*profile["temperature"]) + degradation * 8
            load_pct = clamp(random.uniform(*profile["load"]) + degradation * 5, 0, 100)
            current = clamp(voltage / 12 + load_pct * 0.18 + random.uniform(-2, 2), 0, 120)
            power_kw = voltage * current / 1000
            error_count_24h = int(random.random() < degradation * 0.7) + int(random.random() < degradation * 0.35)
            error_count_7d = error_count_24h + random.randint(0, 4) + int(degradation * 5)

            if profile_name == "critical" and progress > 0.35:
                load_pct = max(load_pct, 88)
                error_count_24h = max(error_count_24h, 5)
                error_count_7d = max(error_count_7d, 12)

            failure = int(
                (
                    profile_name == "monitor"
                    and progress > 0.55
                    and vibration > 4.5
                    and temperature > 78
                    and (error_count_24h >= 1 or load_pct > 72)
                )
                or (
                    profile_name == "critical"
                    and progress > 0.35
                    and vibration > 7
                    and temperature > 90
                    and (error_count_24h >= 2 or load_pct > 85)
                )
            )
            if failure:
                operating_state = "critical"
            elif profile_name == "monitor" or degradation > 0.2:
                operating_state = "degraded"
            else:
                operating_state = "running"

            vibration_delta = 0.0 if previous_vibration is None else vibration - previous_vibration
            pressure_delta = 0.0 if previous_pressure is None else pressure - previous_pressure
            temperature_delta = 0.0 if previous_temperature is None else temperature - previous_temperature

            writer.writerow(
                {
                    "machine_id": profile["machine_id"],
                    "timestamp": timestamp.isoformat(),
                    "voltage": round(voltage, 2),
                    "rpm": round(rpm, 2),
                    "pressure_psi": round(pressure, 2),
                    "vibration_mm_s": round(vibration, 2),
                    "temperature_c": round(temperature, 2),
                    "current": round(current, 2),
                    "power_kw": round(power_kw, 2),
                    "load_pct": round(load_pct, 2),
                    "operating_state": operating_state,
                    "maintenance_count": maintenance_count,
                    "error_count_24h": error_count_24h,
                    "error_count_7d": error_count_7d,
                    "age": age,
                    "vibration_delta": round(vibration_delta, 4),
                    "pressure_delta": round(pressure_delta, 4),
                    "temperature_delta": round(temperature_delta, 4),
                    "failure": failure,
                }
            )

            previous_vibration = vibration
            previous_pressure = pressure
            previous_temperature = temperature
            timestamp += INTERVAL

    print(f"Created {output_path}")


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for profile_name, profile in PROFILES.items():
        generate_machine(profile_name, profile)
    print("Created healthy, monitor, and critical demo telemetry files.")


if __name__ == "__main__":
    main()
