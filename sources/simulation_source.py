from datetime import datetime
import random

from schemas.telemetry import CanonicalTelemetry
from sources.base_source import TelemetrySource


class SimulationSource(TelemetrySource):
    def __init__(self, machine_id="SIM-001", rate_hz=1):
        self.machine_id = machine_id
        self.rate_hz = rate_hz

    def get_event(self) -> CanonicalTelemetry:
        failure_mode = random.random()

        if failure_mode < 0.15:
            # severe failure pattern
            voltage = random.uniform(350, 410)
            rpm = random.uniform(4200, 5200)
            pressure_psi = random.uniform(150, 225)
            vibration_mm_s = random.uniform(9.0, 18.0)
            temperature_c = random.uniform(100, 130)
            current = random.uniform(75, 120)
            power_kw = random.uniform(45, 80)
        elif failure_mode < 0.35:
            # degradation pattern
            voltage = random.uniform(390, 470)
            rpm = random.uniform(3200, 4300)
            pressure_psi = random.uniform(120, 180)
            vibration_mm_s = random.uniform(7.0, 11.0)
            temperature_c = random.uniform(88, 105)
            current = random.uniform(60, 90)
            power_kw = random.uniform(30, 60)
        elif failure_mode < 0.55:
            # unstable operating pattern
            voltage = random.uniform(430, 500)
            rpm = random.uniform(2600, 3900)
            pressure_psi = random.uniform(105, 155)
            vibration_mm_s = random.uniform(5.0, 8.5)
            temperature_c = random.uniform(80, 98)
            current = random.uniform(50, 75)
            power_kw = random.uniform(22, 45)
        else:
            # normal operation
            voltage = random.uniform(450, 520)
            rpm = random.uniform(1800, 3000)
            pressure_psi = random.uniform(80, 120)
            vibration_mm_s = random.uniform(1.5, 4.5)
            temperature_c = random.uniform(55, 78)
            current = random.uniform(20, 45)
            power_kw = random.uniform(8, 24)

        return CanonicalTelemetry(
            machine_id=self.machine_id,
            timestamp=datetime.now(),
            voltage=round(voltage, 2),
            rpm=round(rpm, 2),
            pressure_psi=round(pressure_psi, 2),
            vibration_mm_s=round(vibration_mm_s, 2),
            temperature_c=round(temperature_c, 2),
            current=round(current, 2),
            power_kw=round(power_kw, 2),
            age=random.randint(0, 18),
        )

    def get_stream(self):
        while True:
            yield self.get_event()
