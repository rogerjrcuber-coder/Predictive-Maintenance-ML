import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestProjectPaths(unittest.TestCase):
    def test_app_model_loads_from_project_root(self):
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                spec = importlib.util.spec_from_file_location("predictive_app", ROOT / "app.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                self.assertTrue(hasattr(module, "app"))
                self.assertTrue(Path(module.MODEL_PATH).exists())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
