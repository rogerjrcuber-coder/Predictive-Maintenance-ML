from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from schemas.telemetry import CanonicalTelemetry
from sources.csv_source import CSVSource
from sources.simulation_source import SimulationSource

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ==========================================
# Load Model
# ==========================================

model = joblib.load(MODEL_PATH)
print(type(model))

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Predictive Maintenance Studio",
    description="AI-powered predictive maintenance API",
    version="1.0.0"
)

# ==========================================
# Request Schema
# ==========================================

class PredictionRequest(BaseModel):
    voltage: float
    rpm: float
    pressure: float
    vibration: float
    age: int
    temperature: float = 25.0
    current: float = 0.0
    power_kw: float = 0.0
    load_pct: float = 0.0
    maintenance_count: int = 0
    error_count_24h: int = 0
    error_count_7d: int = 0


class SourceConfig(BaseModel):
    source: str = "simulation"
    machine_id: str = "SIM-001"
    csv_path: str | None = None

# ==========================================
# Health Check
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance Studio API",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# ==========================================
# Risk Calculator
# ==========================================

def calculate_risk(probability):

    if probability >= 0.70:
        return "Critical"

    elif probability >= 0.45:
        return "High"

    elif probability >= 0.20:
        return "Medium"

    else:
        return "Low"

# ==========================================
# Recommendations
# ==========================================

def get_recommendation(risk):

    recommendations = {

        "Critical":
        "Immediate inspection required. Schedule maintenance within 24 hours.",

        "High":
        "Schedule maintenance within 48 hours.",

        "Medium":
        "Monitor closely and inspect during next maintenance window.",

        "Low":
        "Normal operation. Continue monitoring."
    }

    return recommendations[risk]

# ==========================================
# Prediction Endpoint
# ==========================================

def build_source(source_name: str, machine_id: str = "SIM-001", csv_path: str | None = None):
    if source_name == "csv":
        if not csv_path:
            raise ValueError("csv_path is required for csv source")
        return CSVSource(csv_path)
    return SimulationSource(machine_id=machine_id)


@app.post("/predict")
def predict(request: PredictionRequest):
    feature_values = request.model_dump()
    feature_values["pressure_psi"] = request.pressure
    feature_values["vibration_mm_s"] = request.vibration
    feature_values["temperature_c"] = request.temperature
    model_features = list(
        getattr(model, "feature_names_in_", ["voltage", "rpm", "pressure", "vibration", "age"])
    )
    features = np.array(
        [[feature_values.get(name, 0.0) for name in model_features]],
        dtype=float,
    )

    probability = float(
        model.predict_proba(features)[0][1]
    )

    prediction = int(
        probability > 0.5
    )

    risk_level = calculate_risk(probability)

    recommendation = get_recommendation(
        risk_level
    )

    return {
        "raw_probability": probability,
        "percentage": probability * 100,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "prediction": prediction,
    }


@app.post("/ingest")
def ingest(config: SourceConfig):
    source = build_source(config.source, config.machine_id, config.csv_path)
    event = source.get_event()

    model_features = list(getattr(model, "feature_names_in_", ["voltage", "rpm", "pressure", "vibration", "age"]))
    features = np.array([event.to_feature_vector(model_features)], dtype=float)
    probability = float(model.predict_proba(features)[0][1])

    return {
        "source": config.source,
        "machine_id": event.machine_id,
        "timestamp": event.timestamp.isoformat(),
        "raw_probability": probability,
        "percentage": probability * 100,
        "risk_level": calculate_risk(probability),
        "canonical_event": event.model_dump(),
    }