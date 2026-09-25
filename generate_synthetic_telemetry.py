import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "data" / "synthetic" / "synthetic_1M_telemetry.csv"
NUM_MACHINES = 100
ROWS_PER_MACHINE = 10_000
RANDOM_SEED = 42


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def generate():
    random.seed(RANDOM_SEED)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime(2026, 1, 1)

    columns = [
        "machine_id", "timestamp", "voltage", "rpm", "pressure", "vibration",
        "temperature", "current", "power_kw", "load_pct", "operating_state",
        "maintenance_count", "error_count_24h", "error_count_7d", "age",
        "vibration_delta", "pressure_delta", "temperature_delta", "failure",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        for machine_number in range(1, NUM_MACHINES + 1):
            machine_id = f"SIM-{machine_number:03d}"
            voltage = random.uniform(470, 490)
            rpm = random.uniform(2800, 3200)
            pressure = random.uniform(95, 105)
            vibration = random.uniform(1.0, 3.0)
            temperature = random.uniform(65, 75)
            current = random.uniform(25, 45)
            maintenance_count = random.randint(0, 8)
            age = random.randint(1, 20)
            current_time = start_time
            previous_vibration = vibration
            previous_pressure = pressure
            previous_temperature = temperature

            for row_number in range(ROWS_PER_MACHINE):
                degradation = max(0.0, (row_number - 6500) / 3500)
                voltage = random.uniform(470, 490) - degradation * random.uniform(5, 20)
                rpm = random.uniform(2800, 3200) + degradation * random.uniform(200, 700)
                pressure = random.uniform(95, 105) + degradation * random.uniform(10, 45)
                vibration = random.uniform(1.0, 3.0) + degradation * random.uniform(6, 16)
                temperature = random.uniform(65, 75) + degradation * random.uniform(25, 60)
                load_pct = clamp(62 + random.uniform(-8, 8) + degradation * 25, 0, 100)
                current = clamp(25 + load_pct * 0.32 + random.uniform(-3, 3) + degradation * 8, 0, 120)
                power_kw = clamp(voltage * current / 1000 * 0.86, 0, 100)

                error_count_24h = max(0, int(random.expovariate(1 / (0.4 + degradation * 3))))
                error_count_7d = error_count_24h + random.randint(0, 5) + int(degradation * 10)
                operating_state = "degraded" if degradation > 0.5 else "running"

                failure = int(
                    degradation > 0.75
                    and vibration > 7.0
                    and temperature > 90
                    and (error_count_24h >= 2 or load_pct > 85)
                )
                if failure:
                    operating_state = "critical"

                writer.writerow([
                    machine_id,
                    current_time.isoformat(),
                    round(voltage, 2),
                    round(rpm, 1),
                    round(pressure, 2),
                    round(vibration, 2),
                    round(temperature, 2),
                    round(current, 2),
                    round(power_kw, 2),
                    round(load_pct, 2),
                    operating_state,
                    maintenance_count,
                    error_count_24h,
                    error_count_7d,
                    age,
                    round(vibration - previous_vibration, 4),
                    round(pressure - previous_pressure, 4),
                    round(temperature - previous_temperature, 4),
                    failure,
                ])

                previous_vibration = vibration
                previous_pressure = pressure
                previous_temperature = temperature
                current_time += timedelta(minutes=1)

    print(f"Generated {NUM_MACHINES * ROWS_PER_MACHINE:,} rows at {OUTPUT_FILE}")


if __name__ == "__main__":
    generate()