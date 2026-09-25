import json
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


class ModelRegistry:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.registry_path = self.base_dir / "registry.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def fingerprint(path):
        digest = sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def create_run_id(self):
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def save_run(self, run_id, model_path, report):
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, run_dir / "model.pkl")
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        registry = self._read_registry()
        registry["runs"].append(
            {
                "run_id": run_id,
                "created_at": report["created_at"],
                "dataset": report["dataset"],
                "dataset_fingerprint": report["dataset_fingerprint"],
                "status": report["status"],
                "metrics": report["best_metrics"],
                "model_path": str(run_dir / "model.pkl"),
            }
        )
        registry["latest_run_id"] = run_id
        if report["status"] == "accepted":
            registry["production_run_id"] = run_id
        self.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return run_dir

    def _read_registry(self):
        if not self.registry_path.exists():
            return {"latest_run_id": None, "production_run_id": None, "runs": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))
