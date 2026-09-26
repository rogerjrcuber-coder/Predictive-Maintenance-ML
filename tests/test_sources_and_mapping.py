import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents.profiling_agent import ProfilingAgent
from mlops.registry import ModelRegistry
from schemas.telemetry import CanonicalTelemetry
from sources.rolling_demo_source import RollingDemoSource
from sources.schema_mapper import SchemaMapper
from sources.simulation_source import SimulationSource


class TestSourceAdapters(unittest.TestCase):
    def test_schema_mapper_normalizes_aliases(self):
        mapper = SchemaMapper()
        row = {
            "Machine": "PUMP-101",
            "Time": "2026-09-14T12:00:00",
            "PSI": 120,
            "Temp_F": 194,
            "MotorRotation": 3000,
            "Bearing_Vib": 4.5,
            "Humidity": 55,
            "Equipment": "Turbine",
            "Location": "Atlanta",
            "faulty": 0,
            "Load %": 72,
            "Errors 24h": 2,
        }

        canonical = mapper.normalize(row)

        self.assertIsInstance(canonical, CanonicalTelemetry)
        self.assertEqual(canonical.machine_id, "PUMP-101")
        self.assertEqual(canonical.pressure_psi, 120.0)
        self.assertEqual(canonical.temperature_c, 90.0)
        self.assertEqual(canonical.rpm, 3000.0)
        self.assertEqual(canonical.vibration_mm_s, 4.5)
        self.assertEqual(canonical.humidity_pct, 55.0)
        self.assertEqual(canonical.equipment_type, "Turbine")
        self.assertEqual(canonical.location, "Atlanta")
        self.assertEqual(canonical.faulty, 0)
        self.assertEqual(canonical.load_pct, 72.0)
        self.assertEqual(canonical.error_count_24h, 2)

    def test_simulation_source_produces_canonical_event(self):
        source = SimulationSource(machine_id="PUMP-200")
        event = source.get_event()

        self.assertIsInstance(event, CanonicalTelemetry)
        self.assertEqual(event.machine_id, "PUMP-200")
        self.assertGreater(event.voltage, 0)
        self.assertGreater(event.rpm, 0)

    def test_schema_mapper_normalizes_punctuation_and_units(self):
        mapper = SchemaMapper()
        row = {
            "Machine ID": "PUMP-202",
            "Event-Time": "2026-09-14T12:00:00",
            "Supply Voltage": 480,
            "Rot-Speed": 3000,
            "Pressure kPa": 689.475,
            "Temp F": 194,
            "Vibe": 4.5,
        }

        canonical = mapper.normalize(row)

        self.assertEqual(canonical.machine_id, "PUMP-202")
        self.assertAlmostEqual(canonical.pressure_psi, 100.0, places=2)
        self.assertAlmostEqual(canonical.temperature_c, 90.0, places=2)
        self.assertEqual(canonical.rpm, 3000.0)

    def test_profiling_agent_reports_imbalance(self):
        dataframe = pd.DataFrame({"voltage": [480, 481, 482], "failure": [0, 0, 1]})

        report = ProfilingAgent().profile(dataframe)

        self.assertEqual(report.rows, 3)
        self.assertEqual(report.columns, 2)
        self.assertEqual(report.missing_values, 0)
        self.assertGreater(report.class_imbalance_ratio, 1)
        self.assertEqual(report.recommended_model, "Random Forest")

    def test_model_registry_fingerprints_and_writes_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.csv"
            model = root / "model.pkl"
            dataset.write_text("failure\n0\n1\n", encoding="utf-8")
            model.write_bytes(b"model")
            registry = ModelRegistry(root / "models")
            report = {
                "created_at": "2026-09-22T00:00:00+00:00",
                "dataset": str(dataset),
                "dataset_fingerprint": registry.fingerprint(dataset),
                "status": "accepted",
                "best_metrics": {"roc_auc": 0.9, "failure_recall": 0.8},
            }

            run_dir = registry.save_run("run-1", model, report)

            self.assertTrue((run_dir / "model.pkl").exists())
            self.assertTrue((run_dir / "report.json").exists())
            self.assertEqual(registry._read_registry()["production_run_id"], "run-1")

    def test_rolling_demo_source_appends_only_csv_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "demo.csv"
            csv_path.write_text(
                "machine_id,timestamp,voltage,rpm,pressure_psi,vibration_mm_s,temperature_c,current,power_kw,load_pct,operating_state,maintenance_count,error_count_24h,error_count_7d,age,vibration_delta,pressure_delta,temperature_delta,failure\n"
                "PUMP-TEST,2026-09-01T00:00:00,480,3000,100,2.5,70,40,20,50,running,0,0,0,5,0,0,0,0\n",
                encoding="utf-8",
            )

            source = RollingDemoSource.__new__(RollingDemoSource)
            source.scenario = "Healthy"
            source.file_path = csv_path

            source.append_event()

            lines = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("machine_id", lines[0])
            self.assertNotIn("faulty", lines[0])

    def test_risk_calculation_includes_high_level(self):
        from app import calculate_risk

        self.assertEqual(calculate_risk(0.9), "Critical")
        self.assertEqual(calculate_risk(0.75), "Critical")
        self.assertEqual(calculate_risk(0.60), "High")
        self.assertEqual(calculate_risk(0.40), "Medium")
        self.assertEqual(calculate_risk(0.10), "Low")

    def test_predict_accepts_five_feature_request_with_wider_model(self):
        from app import PredictionRequest, predict

        result = predict(
            PredictionRequest(
                voltage=480,
                rpm=3000,
                pressure=100,
                vibration=2.5,
                age=5,
            )
        )

        self.assertIn("raw_probability", result)
        self.assertGreaterEqual(result["raw_probability"], 0.0)
        self.assertLessEqual(result["raw_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
