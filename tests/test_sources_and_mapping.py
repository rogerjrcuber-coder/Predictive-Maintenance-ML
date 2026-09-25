import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents.profiling_agent import ProfilingAgent
from mlops.registry import ModelRegistry
from schemas.telemetry import CanonicalTelemetry
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


if __name__ == "__main__":
    unittest.main()
