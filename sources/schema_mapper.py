from datetime import datetime
import re

from schemas.telemetry import CanonicalTelemetry


class SchemaMapper:
    alias_map = {
        "machine_id": ["machine_id", "machine", "machineID", "asset_id", "equipment_id", "unit_id", "id"],
        "timestamp": ["timestamp", "datetime", "time", "ts", "date_time", "event_time"],
        "voltage": ["voltage", "volt", "volts", "voltage_sensor", "supply_voltage", "line_voltage", "electrical_voltage"],
        "rpm": ["rpm", "rotate", "rotation", "rotationalspeed", "rot_speed", "motor_rpm", "motor_speed", "shaft_speed", "rotational_speed", "speed_revolutions", "motorrotation"],
        "pressure_psi": ["pressure_psi", "pressure", "psi", "press", "pres", "pressure_sensor", "pump_pressure", "line_pressure", "process_pressure", "pres_pump", "pressure_kpa", "pressure_bar"],
        "vibration_mm_s": ["vibration_mm_s", "vibration", "vib", "vibe", "bearing_vibration", "oscillation", "vibration_velocity", "bearing_vib", "vibration_sensor"],
        "temperature_c": ["temperature_c", "temperature", "temp", "temp_c", "motor_temp", "bearing_temp", "ambient_temp", "thermal", "temperature_sensor", "temp_f"],
        "humidity_pct": ["humidity_pct", "humidity", "relative_humidity", "rh", "humidity_percent"],
        "equipment_type": ["equipment_type", "equipment", "machine_type", "asset_type"],
        "location": ["location", "site", "facility", "plant", "region"],
        "current": ["current", "amp", "amps", "amperage", "current_draw", "motor_current", "electrical_current"],
        "power_kw": ["power_kw", "power", "kw", "kilowatts", "motor_power", "load_power"],
        "load_pct": ["load_pct", "load", "load_percent", "load_percentage", "utilization"],
        "operating_state": ["operating_state", "state", "status", "machine_state", "operational_state"],
        "maintenance_count": ["maintenance_count", "maintenance_events", "service_count", "repairs"],
        "error_count_24h": ["error_count_24h", "errors_24h", "error_count_last_24h", "recent_errors"],
        "error_count_7d": ["error_count_7d", "errors_7d", "error_count_last_7d", "weekly_errors"],
        "vibration_delta": ["vibration_delta", "vibration_change", "vibration_trend"],
        "pressure_delta": ["pressure_delta", "pressure_change", "pressure_trend"],
        "temperature_delta": ["temperature_delta", "temperature_change", "temperature_trend"],
        "faulty": ["faulty", "fault", "failure", "failed", "anomaly", "label"],
        "age": ["age", "machine_age", "asset_age", "service_age"],
    }

    def normalize(self, row):
        mapped = {}
        matched_aliases = self.map_columns(row.keys())

        for canonical_name, aliases in self.alias_map.items():
            source_name = matched_aliases.get(canonical_name)
            value = row[source_name] if source_name is not None else None

            if value is None:
                continue

            if canonical_name == "timestamp":
                mapped[canonical_name] = self._to_datetime(value)
            elif canonical_name == "temperature_c" and isinstance(value, (int, float)) and self._normalize_name(source_name) == "tempf":
                mapped[canonical_name] = self._fahrenheit_to_celsius(float(value))
            elif canonical_name == "pressure_psi" and isinstance(value, (int, float)) and self._normalize_name(source_name).endswith("kpa"):
                mapped[canonical_name] = float(value) * 0.145038
            elif canonical_name == "pressure_psi" and isinstance(value, (int, float)) and self._normalize_name(source_name).endswith("bar"):
                mapped[canonical_name] = float(value) * 14.5038
            elif canonical_name == "pressure_psi" and isinstance(value, (int, float)):
                mapped[canonical_name] = float(value)
            else:
                mapped[canonical_name] = value

        machine_id = mapped.get("machine_id") or row.get("machine_id") or row.get("Machine") or "UNKNOWN"
        timestamp = mapped.get("timestamp") or datetime.now()
        voltage = float(mapped.get("voltage", 0.0) or 0.0)
        rpm = float(mapped.get("rpm", 0.0) or 0.0)
        pressure_psi = float(mapped.get("pressure_psi", 0.0) or 0.0)
        vibration_mm_s = float(mapped.get("vibration_mm_s", 0.0) or 0.0)
        temperature_c = float(mapped.get("temperature_c", 25.0) or 25.0)
        humidity_pct = float(mapped.get("humidity_pct", 0.0) or 0.0)
        equipment_type = str(mapped.get("equipment_type", "unknown") or "unknown")
        location = str(mapped.get("location", "unknown") or "unknown")
        current = float(mapped.get("current", 0.0) or 0.0)
        power_kw = float(mapped.get("power_kw", 0.0) or 0.0)
        load_pct = float(mapped.get("load_pct", 0.0) or 0.0)
        operating_state = str(mapped.get("operating_state", "running") or "running")
        maintenance_count = int(mapped.get("maintenance_count", 0) or 0)
        error_count_24h = int(mapped.get("error_count_24h", 0) or 0)
        error_count_7d = int(mapped.get("error_count_7d", 0) or 0)
        vibration_delta = float(mapped.get("vibration_delta", 0.0) or 0.0)
        pressure_delta = float(mapped.get("pressure_delta", 0.0) or 0.0)
        temperature_delta = float(mapped.get("temperature_delta", 0.0) or 0.0)
        faulty_value = mapped.get("faulty")
        faulty = None if faulty_value is None else int(float(faulty_value))
        age = int(mapped.get("age", 0) or 0)

        return CanonicalTelemetry(
            machine_id=str(machine_id),
            timestamp=timestamp,
            voltage=voltage,
            rpm=rpm,
            pressure_psi=pressure_psi,
            vibration_mm_s=vibration_mm_s,
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            equipment_type=equipment_type,
            location=location,
            current=current,
            power_kw=power_kw,
            load_pct=load_pct,
            operating_state=operating_state,
            maintenance_count=maintenance_count,
            error_count_24h=error_count_24h,
            error_count_7d=error_count_7d,
            vibration_delta=vibration_delta,
            pressure_delta=pressure_delta,
            temperature_delta=temperature_delta,
            faulty=faulty,
            age=age,
        )

    def map_columns(self, columns):
        normalized_columns = {
            self._normalize_name(column): column for column in columns
        }
        matches = {}

        for canonical_name, aliases in self.alias_map.items():
            for alias in [canonical_name, *aliases]:
                normalized_alias = self._normalize_name(alias)
                if normalized_alias in normalized_columns:
                    matches[canonical_name] = normalized_columns[normalized_alias]
                    break

        return matches

    @staticmethod
    def _normalize_name(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    @staticmethod
    def _to_datetime(value):
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

    @staticmethod
    def _fahrenheit_to_celsius(value):
        return (value - 32) * 5 / 9
