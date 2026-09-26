import csv
import random
from datetime import timedelta
from pathlib import Path

from schemas.telemetry import CanonicalTelemetry
from sources.csv_source import CSVSource


BASE_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = BASE_DIR / "data" / "synthetic" / "demo"
SCENARIO_FILES = {
    "Healthy": DEMO_DIR / "healthy_machine.csv",
    "Medium": DEMO_DIR / "monitor_machine.csv",
    "Critical": DEMO_DIR / "critical_machine.csv",
}
CSV_COLUMNS = [
    "machine_id", "timestamp", "voltage", "rpm", "pressure_psi",
    "vibration_mm_s", "temperature_c", "current", "power_kw", "load_pct",
    "operating_state", "maintenance_count", "error_count_24h", "error_count_7d",
    "age", "vibration_delta", "pressure_delta", "temperature_delta", "failure",
]


class RollingDemoSource:
    def __init__(self, scenario):
        if scenario not in SCENARIO_FILES:
            raise ValueError(f"Unknown rolling scenario: {scenario}")
        self.scenario = scenario
        self.file_path = SCENARIO_FILES[scenario]

    def append_event(self):
        previous = CSVSource(self.file_path).get_event()
        row_count = self.row_count()
        progress = min(1.0, max(0.0, row_count / 1000))
        event, failure = self._next_event(previous, progress)
        self._append(event, failure)
        return event

    def row_count(self):
        if not self.file_path.exists():
            return 0
        with self.file_path.open("r", encoding="utf-8") as file:
            return max(0, sum(1 for _ in file) - 1)

    def _next_event(self, previous, progress):
        if self.scenario == "Healthy":
            vibration = random.uniform(1.0, 3.0)
            temperature = random.uniform(60.0, 70.0)
            load = random.uniform(35.0, 60.0)
            errors_24h = 0
            state = "running"
            failure = 0
        elif self.scenario == "Medium":
            vibration = random.uniform(3.0, 6.5) + progress * 1.5
            temperature = random.uniform(70.0, 85.0) + progress * 6
            load = random.uniform(55.0, 80.0) + progress * 5
            errors_24h = int(random.random() < 0.3 + progress * 0.4)
            state = "degraded"
            failure = int(progress > 0.55 and vibration > 4.5 and temperature > 78)
        else:
            vibration = random.uniform(8.0, 14.0) + progress * 2
            temperature = random.uniform(92.0, 115.0) + progress * 8
            load = max(88.0, random.uniform(85.0, 98.0))
            errors_24h = max(5, int(random.uniform(5, 10)))
            state = "critical"
            failure = int(vibration > 7 and temperature > 90)

        voltage = random.uniform(470, 490) - progress * (8 if self.scenario != "Healthy" else 1)
        rpm = random.uniform(2900, 3100) + progress * (250 if self.scenario != "Healthy" else 20)
        pressure = random.uniform(95, 105) + progress * (25 if self.scenario != "Healthy" else 3)
        current = max(0.0, voltage / 12 + load * 0.18 + random.uniform(-2, 2))
        power_kw = voltage * current / 1000
        timestamp = previous.timestamp + timedelta(minutes=5)

        event = CanonicalTelemetry(
            machine_id=previous.machine_id,
            timestamp=timestamp,
            voltage=round(voltage, 2),
            rpm=round(rpm, 2),
            pressure_psi=round(pressure, 2),
            vibration_mm_s=round(vibration, 2),
            temperature_c=round(temperature, 2),
            current=round(current, 2),
            power_kw=round(power_kw, 2),
            load_pct=round(min(load, 100), 2),
            operating_state=state,
            maintenance_count=previous.maintenance_count,
            error_count_24h=errors_24h,
            error_count_7d=max(errors_24h, previous.error_count_7d),
            age=previous.age,
            vibration_delta=round(vibration - previous.vibration_mm_s, 4),
            pressure_delta=round(pressure - previous.pressure_psi, 4),
            temperature_delta=round(temperature - previous.temperature_c, 4),
        )
        return event, failure

    def _append(self, event, failure):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.file_path.exists() and self.file_path.stat().st_size > 0
        with self.file_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            if not exists:
                writer.writeheader()

            row = {
                "machine_id": event.machine_id,
                "timestamp": event.timestamp.isoformat(),
                "voltage": event.voltage,
                "rpm": event.rpm,
                "pressure_psi": event.pressure_psi,
                "vibration_mm_s": event.vibration_mm_s,
                "temperature_c": event.temperature_c,
                "current": event.current,
                "power_kw": event.power_kw,
                "load_pct": event.load_pct,
                "operating_state": event.operating_state,
                "maintenance_count": event.maintenance_count,
                "error_count_24h": event.error_count_24h,
                "error_count_7d": event.error_count_7d,
                "age": event.age,
                "vibration_delta": event.vibration_delta,
                "pressure_delta": event.pressure_delta,
                "temperature_delta": event.temperature_delta,
                "failure": failure,
            }
            writer.writerow(row)
