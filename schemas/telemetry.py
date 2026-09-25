from datetime import datetime

from pydantic import BaseModel, Field


class CanonicalTelemetry(BaseModel):
    machine_id: str
    timestamp: datetime
    voltage: float = Field(..., ge=0)
    rpm: float = Field(..., ge=0)
    pressure_psi: float = Field(..., ge=0)
    vibration_mm_s: float = Field(..., ge=0)
    temperature_c: float = Field(..., ge=-100)
    humidity_pct: float = Field(default=0.0, ge=0, le=100)
    equipment_type: str = "unknown"
    location: str = "unknown"
    current: float = Field(default=0.0, ge=0)
    power_kw: float = Field(default=0.0, ge=0)
    load_pct: float = Field(default=0.0, ge=0, le=100)
    operating_state: str = "running"
    maintenance_count: int = Field(default=0, ge=0)
    error_count_24h: int = Field(default=0, ge=0)
    error_count_7d: int = Field(default=0, ge=0)
    vibration_delta: float = 0.0
    pressure_delta: float = 0.0
    temperature_delta: float = 0.0
    faulty: int | None = None
    age: int = Field(default=0, ge=0)

    @property
    def feature_vector(self):
        return self.to_feature_vector(["voltage", "rpm", "pressure", "vibration", "age"])

    def to_feature_vector(self, feature_names):
        values = {
            "voltage": self.voltage,
            "rpm": self.rpm,
            "pressure": self.pressure_psi,
            "pressure_psi": self.pressure_psi,
            "vibration": self.vibration_mm_s,
            "vibration_mm_s": self.vibration_mm_s,
            "temperature": self.temperature_c,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "current": self.current,
            "power_kw": self.power_kw,
            "load_pct": self.load_pct,
            "maintenance_count": self.maintenance_count,
            "error_count_24h": self.error_count_24h,
            "error_count_7d": self.error_count_7d,
            "vibration_delta": self.vibration_delta,
            "pressure_delta": self.pressure_delta,
            "temperature_delta": self.temperature_delta,
            "age": self.age,
        }
        return [values.get(name, 0.0) for name in feature_names]
