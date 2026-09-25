from pathlib import Path

import pandas as pd
import streamlit as st

from app import calculate_risk, get_recommendation, model
from sources.csv_source import CSVSource
from sources.rolling_demo_source import RollingDemoSource
from sources.simulation_source import SimulationSource


BASE_DIR = Path(__file__).resolve().parent
TRAINING_REPORT_PATH = BASE_DIR / "models" / "training_report.json"

st.set_page_config(
    page_title="mAIntAIn | Predictive Maintenance",
    page_icon="⚙",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #181a1b; color: #e8e6e3; }
    .block-container { max-width: 1280px; padding-top: 2rem; }
    [data-testid="stSidebar"] { background: #202324; border-right: 1px solid #3a3f41; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #e8e6e3; }
    .hero { padding: 1.25rem 0 1rem; border-bottom: 1px solid #3a3f41; }
    .wordmark { font-family: Arial, Helvetica, sans-serif; font-size: 3rem; font-style: italic; font-weight: 900; letter-spacing: -0.12em; line-height: 1; }
    .wordmark .white { color: #ffffff; }
    .wordmark .orange { color: #f28c00; }
    .eyebrow { color: #66c7d9; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; }
    .hero p, .stCaption { color: #a9a9a9; margin-bottom: 0; }
    [data-testid="stMetric"] { background: #202324; border: 1px solid #3a3f41; padding: 1rem; border-radius: 0.35rem; }
    [data-testid="stMetricLabel"] { color: #a9a9a9; }
    [data-testid="stMetricValue"] { color: #f28c00; }
    .stTextInput input, [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] { background: #181a1b; color: #e8e6e3; border-color: #4a5052; }
    .stButton button { background: #f28c00; color: #181a1b; border: 0; font-weight: 700; }
    .stButton button:hover { background: #ffad33; color: #181a1b; }
    [data-testid="stDataFrame"] { border: 1px solid #3a3f41; }
    .agent { border-left: 4px solid #2e9d78; background: #202b28; color: #d9f2e8; padding: 0.6rem 0.8rem; margin: 0.4rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
            <div class="wordmark" aria-label="mAIntAIn"><span class="white">m</span><span class="orange">A</span><span class="orange">I</span><span class="white">n</span><span class="white">t</span><span class="orange">A</span><span class="orange">I</span><span class="white">n</span></div>
            <div class="eyebrow">MULTIPLE AI AGENTS FOR INTELLIGENT MAINTENANCE</div>
      <p>Source-agnostic telemetry, risk scoring, and maintenance decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Telemetry source")
    source_name = st.radio("Choose source", ["Simulation", "CSV", "Rolling demo"], horizontal=True)
    machine_id = st.text_input("Machine ID", "SIM-001")
    csv_path = None
    rolling_scenario = "Healthy"

    if source_name == "CSV":
        uploaded_file = st.file_uploader("Upload telemetry CSV", type=["csv"])
        if uploaded_file is not None:
            csv_path = uploaded_file
        else:
            st.caption("Upload a CSV with recognizable telemetry column names.")
    elif source_name == "Rolling demo":
        rolling_scenario = st.selectbox(
            "Scenario",
            ["Healthy", "Medium", "Critical"],
            help="Each generation appends one row to the scenario CSV.",
        )
        refresh_interval = st.slider(
            "Generation interval (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=5.0,
            step=0.5,
        )
        st.caption("Each refresh appends one new telemetry row to the selected CSV.")
        auto_generate = st.checkbox("Auto-generate automatically", value=False)
    else:
        refresh_interval = 5.0
        auto_generate = False

    generate = st.button("Generate telemetry", type="primary", use_container_width=True)

    if TRAINING_REPORT_PATH.exists():
        import json

        training_report = json.loads(TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
        st.divider()
        st.caption("MLOps model registry")
        st.caption(f"Run: {training_report.get('run_id', 'unknown')}")
        st.caption(f"Dataset: {training_report.get('dataset_name', 'unknown')}")
        metrics = training_report.get("best_metrics", {})
        st.caption(
            f"ROC-AUC {metrics.get('roc_auc', 0):.3f} | "
            f"Recall {metrics.get('failure_recall', 0):.3f}"
        )


def load_event():
    if source_name == "CSV":
        if csv_path is None:
            st.warning("Upload a CSV file before generating telemetry.")
            return None, None
        source = CSVSource(csv_path)
        with st.spinner("Processing the latest telemetry window..."):
            events = source.get_recent_events(window_size=100)
        if not events:
            st.warning("The CSV file does not contain telemetry rows.")
            return None, None
        model_features = list(getattr(model, "feature_names_in_", ["voltage", "rpm", "pressure", "vibration", "age"]))
        probabilities = model.predict_proba(
            [event.to_feature_vector(model_features) for event in events]
        )[:, 1]
        window_size = min(100, len(probabilities))
        return events[-1], float(probabilities[-window_size:].mean())
    elif source_name == "Rolling demo":
        source = RollingDemoSource(rolling_scenario)
        event = source.append_event()
        model_features = list(getattr(model, "feature_names_in_", ["voltage", "rpm", "pressure", "vibration", "age"]))
        latest_probability = float(
            model.predict_proba([event.to_feature_vector(model_features)])[0][1]
        )
        latest_signal = telemetry_signal_score(event)
        latest_risk = max(latest_probability, latest_signal)
        if latest_risk >= 0.70:
            lookback = 200
        elif latest_risk >= 0.30:
            lookback = 150
        else:
            lookback = 100

        recent_events = CSVSource(source.file_path).get_recent_events(window_size=lookback)
        model_probabilities = model.predict_proba(
            [recent_event.to_feature_vector(model_features) for recent_event in recent_events]
        )[:, 1]
        signal_scores = [telemetry_signal_score(recent_event) for recent_event in recent_events]
        probability = float(max(model_probabilities.mean(), sum(signal_scores) / len(signal_scores)))
        return event, probability
    else:
        source = SimulationSource(machine_id=machine_id or "SIM-001")
        event = source.get_event()
        model_features = list(getattr(model, "feature_names_in_", ["voltage", "rpm", "pressure", "vibration", "age"]))
        probability = float(model.predict_proba([event.to_feature_vector(model_features)])[0][1])
        return event, probability


def telemetry_signal_score(event):
    vibration_score = (event.vibration_mm_s - 2.0) / 10.0
    temperature_score = (event.temperature_c - 65.0) / 45.0
    load_score = (event.load_pct - 45.0) / 45.0
    error_score = event.error_count_24h / 5.0
    score = (
        0.35 * vibration_score
        + 0.30 * temperature_score
        + 0.20 * load_score
        + 0.15 * error_score
    )
    return max(0.0, min(1.0, score))


def telemetry_frame(event):
    unit_map = {
        "voltage": "V",
        "rpm": "rpm",
        "pressure_psi": "psi",
        "vibration_mm_s": "mm/s",
        "temperature_c": "C",
        "humidity_pct": "%",
        "current": "A",
        "power_kw": "kW",
        "load_pct": "%",
        "age": "years",
        "maintenance_count": "events",
        "error_count_24h": "events",
        "error_count_7d": "events",
        "vibration_delta": "mm/s",
        "pressure_delta": "psi",
        "temperature_delta": "C",
    }
    label_map = {
        "machine_id": "Machine ID",
        "timestamp": "Timestamp",
        "pressure_psi": "Pressure",
        "vibration_mm_s": "Vibration",
        "temperature_c": "Temperature",
        "humidity_pct": "Humidity",
        "load_pct": "Load",
        "operating_state": "Operating state",
        "maintenance_count": "Maintenance count",
        "error_count_24h": "Errors, 24h",
        "error_count_7d": "Errors, 7d",
    }
    rows = []
    for name, value in event.model_dump().items():
        if name in {"machine_id", "timestamp"}:
            continue
        rows.append(
            {
                "Measurement": label_map.get(name, name.replace("_", " ").title()),
                "Value": value,
                "Unit": unit_map.get(name, ""),
            }
        )
    return pd.DataFrame(rows)


def render_dashboard():
    should_generate = generate or (source_name == "Rolling demo" and auto_generate)
    if should_generate:
        event, probability = load_event()
        if event is not None:
            risk = calculate_risk(probability)
            recommendation = get_recommendation(risk)
            st.session_state["result"] = {
                "event": event,
                "probability": probability,
                "risk": risk,
                "recommendation": recommendation,
            }

    result = st.session_state.get("result")
    if result is None:
        st.info("Choose a source and generate telemetry to inspect a machine.")
        return

    event = result["event"]
    probability = result["probability"]
    risk = result["risk"]

    st.caption(f"Source: {source_name}  |  Machine: {event.machine_id}  |  Timestamp: {event.timestamp:%Y-%m-%d %H:%M:%S}")
    if source_name == "Rolling demo":
        rolling_source = RollingDemoSource(rolling_scenario)
        with rolling_source.file_path.open("r", encoding="utf-8") as file:
            row_count = max(0, sum(1 for _ in file) - 1)
        st.caption(f"Rolling file: {rolling_source.file_path.name}  |  Rows: {row_count}  |  New row interval: 5 seconds")

    col1, col2, col3 = st.columns(3)
    col1.metric("Failure probability", f"{probability:.1%}")
    col2.metric("Risk level", risk)
    col3.metric("Machine age", f"{event.age} years")

    st.subheader("Current machine status")
    status_col, recommendation_col = st.columns([1, 2])
    with status_col:
        if risk == "Critical":
            st.error("Critical attention required")
        elif risk == "High":
            st.warning("High maintenance risk")
        elif risk == "Medium":
            st.warning("Monitor closely")
        else:
            st.success("Operating normally")
    with recommendation_col:
        st.markdown(f"**Recommendation:** {result['recommendation']}")

    st.subheader("Live telemetry")
    st.dataframe(telemetry_frame(event), hide_index=True, use_container_width=True)

    st.subheader("Agent activity")
    for label in [
        "Schema Mapping Agent: canonical telemetry created",
        "Profiling Agent: source event inspected",
        "Prediction Agent: failure probability calculated",
        "Explainability Agent: telemetry drivers available for review",
    ]:
        st.markdown(f'<div class="agent">✓ {label}</div>', unsafe_allow_html=True)


if hasattr(st, "fragment"):
    @st.fragment(run_every=(refresh_interval if source_name == "Rolling demo" and auto_generate else None))
    def live_dashboard():
        render_dashboard()

    live_dashboard()
else:
    render_dashboard()
