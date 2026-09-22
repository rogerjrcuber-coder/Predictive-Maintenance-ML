# mAIntAIn

### Multiple AI Agents for Intelligent Maintenance

An AI-powered predictive maintenance platform that automatically profiles industrial telemetry, harmonizes machine schemas, trains machine learning models, explains equipment failures, and deploys production-ready maintenance services through a team of specialized AI agents.

---

## 🚀 Overview

mAIntAIn is an Agentic Predictive Maintenance Studio designed to simplify the adoption of predictive maintenance in industrial environments.

The platform enables engineers to ingest telemetry from multiple sources, standardize machine data into a common format, train predictive models, analyze equipment health, explain predictions, and deploy production-ready APIs.

The architecture is source-agnostic, allowing simulation data, historical datasets, and future IoT streams to flow through the same prediction pipeline.

---

## 🎯 Project Goals

- Predict equipment failures before they occur
- Automate dataset profiling and quality assessment
- Support multiple telemetry sources
- Provide explainable AI predictions
- Simplify AutoML workflows
- Support MLOps and deployment automation
- Demonstrate a true multi-agent architecture

---

## 💡 Core Innovation

### Schema Harmonization Layer

Industrial telemetry is rarely standardized.

Different equipment manufacturers often report the same measurement under different names:

| Source A | Source B | Canonical Field |
|-----------|-----------|----------------|
| Pressure | PSI | pressure_psi |
| Temp_F | Temperature | temperature_c |
| MotorRotation | RPM | rpm |
| Bearing_Vib | Vibration | vibration_mm_s |

mAIntAIn automatically maps these fields into a unified telemetry schema, allowing different data sources to share the same AI pipeline.

---

## 🏗 Architecture

```text
Real Dataset / Simulation / CSV / IoT Stream
                    |
                    v
          Schema Mapping Layer
                    |
                    v
        CanonicalTelemetry Schema
                    |
                    v
          Feature Engineering Layer
                    |
                    v
             Prediction Model
                    |
                    v
      Risk + Recommendation + Explanation
```

---

## 🤖 AI Agent Team

### 1. Profiling Agent

Responsible for dataset quality assessment.

#### Capabilities

- Missing value detection
- Duplicate identification
- Outlier detection
- Class imbalance analysis
- Data-quality scoring

#### Example Output

```json
{
  "quality_score": 94,
  "duplicates": 0,
  "outliers": 47,
  "recommendation": "Use XGBoost"
}
```

---

### 2. Schema Mapping Agent

Responsible for harmonizing telemetry from multiple manufacturers.

#### Example

```text
Pressure
PSI
PumpPressure
PRES_PUMP
```

↓

```text
pressure_psi
```

---

### 3. Feature Engineering Agent

Generates advanced ML features such as:

- Rolling averages
- Pressure trends
- Temperature drift
- RPM volatility
- Vibration degradation metrics

---

### 4. AutoML Agent

Evaluates multiple models and selects the best performer.

#### Candidate Models

- XGBoost
- LightGBM
- Random Forest
- Gradient Boosting

#### Evaluation Metrics

- ROC-AUC
- PR-AUC
- Recall
- F1 Score

---

### 5. Prediction Agent

Generates:

- Failure probability
- Asset risk score
- Operational status

#### Example Output

```json
{
  "failure_probability": 84.2,
  "risk_level": "High"
}
```

---

### 6. Explainability Agent

Uses SHAP to explain predictions.

#### Example Output

```text
Top Risk Factors

1. Vibration
2. Pressure
3. Machine Age
```

---

### 7. Deployment Agent

Automates deployment workflows.

#### Responsibilities

- FastAPI deployment
- Docker image generation
- MLflow integration
- Model version management

---

## 📊 Supported Data Sources

### Current

- Microsoft Predictive Maintenance Dataset
- Synthetic Telemetry Dataset
- CSV Uploads
- Digital Twin Simulation

### Planned

- NASA CMAPSS Dataset
- MQTT Streams
- Kafka Streams
- Azure IoT Hub
- REST APIs
- Industrial PLC Telemetry

---

## 📦 Canonical Telemetry Schema

All telemetry is converted into a common representation:

```python
CanonicalTelemetry
```

### Fields

```text
machine_id
timestamp
voltage
rpm
pressure_psi
vibration_mm_s
temperature_c
current
power_kw
age
```

This allows the prediction model to remain independent of the original data source.

---

## 🏭 Digital Twin Simulator

The platform includes a built-in simulation system capable of generating:

- Healthy machine behavior
- Warning conditions
- Critical equipment failures

### Benefits

- No physical hardware required
- Repeatable demonstrations
- Realistic telemetry generation
- Seamless migration to real factory streams

---

## 🔧 Technology Stack

### Backend

- Python
- FastAPI
- Pydantic

### Machine Learning

- XGBoost
- LightGBM
- Scikit-Learn
- SHAP

### Agent Framework

- LangGraph

### MLOps

- MLflow
- Docker

### Database

- PostgreSQL

### Dashboard

- Streamlit
- Plotly

---

## 🛣 Roadmap

### Prototype Phase (September 27)

#### Core Features

- [x] Source-Agnostic Ingestion
- [x] Simulation Source
- [x] CSV Source
- [x] Microsoft Dataset Integration
- [x] Canonical Telemetry Schema
- [x] Schema Mapping Layer
- [ ] Profiling Agent
- [ ] AutoML Agent
- [ ] SHAP Explainability
- [ ] Streamlit Dashboard
- [ ] MLflow Tracking

---

### Future Enhancements

- NASA CMAPSS Integration
- Real-Time IoT Streaming
- Advanced LangGraph Orchestration
- Automated Hyperparameter Optimization
- Remaining Useful Life Prediction
- Fleet-Wide Health Analytics
- Azure Deployment Support

---

## 📈 Desired Prediction Flow

```text
Simulation / CSV / NASA / Microsoft / IoT
                    |
                    v
          Schema Mapping Layer
                    |
                    v
        Canonical Telemetry Format
                    |
                    v
          Feature Engineering
                    |
                    v
             Prediction Model
                    |
                    v
      Risk + Explanation + Action
```

---

## ⚡ One-Sentence Pitch

> **mAIntAIn is an AI-powered AutoML and MLOps copilot that automatically understands industrial telemetry, predicts equipment failures, explains root causes, and deploys production-ready predictive maintenance services through a team of specialized AI agents.**
