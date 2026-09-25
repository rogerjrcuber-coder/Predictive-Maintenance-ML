import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from mlops.registry import ModelRegistry


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "microsoft_training.csv"
DATASETS = {
    "microsoft": DATA_PATH,
    "synthetic": BASE_DIR / "data" / "synthetic" / "synthetic_1M_telemetry.csv",
}
DEMO_DATASETS = [
    BASE_DIR / "data" / "synthetic" / "demo" / "healthy_machine.csv",
    BASE_DIR / "data" / "synthetic" / "demo" / "monitor_machine.csv",
    BASE_DIR / "data" / "synthetic" / "demo" / "critical_machine.csv",
]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
REPORT_PATH = BASE_DIR / "models" / "training_report.json"
REGISTRY_DIR = BASE_DIR / "models"
BASE_FEATURES = ["voltage", "rpm", "pressure", "vibration", "age"]
OPTIONAL_FEATURES = [
    "temperature",
    "current",
    "power_kw",
    "load_pct",
    "maintenance_count",
    "error_count_24h",
    "error_count_7d",
]
TARGET_ALIASES = ["failure", "fail", "failure_flag", "failed", "faulty", "label"]

TARGET_ROC_AUC = 0.80
TARGET_RECALL = 0.80
MAX_TRIALS = 6


def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "failure_recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def is_accepted(metrics):
    return metrics["roc_auc"] >= TARGET_ROC_AUC and metrics["failure_recall"] >= TARGET_RECALL


def resolve_dataset(dataset):
    if dataset == "demo":
        missing = [path for path in DEMO_DATASETS if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Demo datasets not found: {missing}")
        return DEMO_DATASETS
    path = DATASETS.get(dataset, Path(dataset))
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        available = ", ".join(DATASETS)
        raise FileNotFoundError(
            f"Dataset not found: {path}. Choose one of: {available}, or provide a CSV path."
        )
    return path


def resolve_target_column(dataframe):
    for column in TARGET_ALIASES:
        if column in dataframe.columns:
            return column
    expected = ", ".join(TARGET_ALIASES)
    raise ValueError(f"Dataset must contain one target column: {expected}")


def resolve_features(dataframe):
    features = [column for column in [*BASE_FEATURES, *OPTIONAL_FEATURES] if column in dataframe]
    missing_base = [column for column in BASE_FEATURES if column not in features]
    if missing_base:
        raise ValueError(f"Dataset is missing required base features: {', '.join(missing_base)}")
    return features


def train(dataset="microsoft"):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    data_path = resolve_dataset(dataset)
    if dataset == "demo":
        frames = []
        for path in data_path:
            frame = pd.read_csv(path).rename(
                columns={
                    "pressure_psi": "pressure",
                    "vibration_mm_s": "vibration",
                    "temperature_c": "temperature",
                }
            )
            frames.append(frame)
        dataframe = pd.concat(frames, ignore_index=True)
        report_dataset = [str(path) for path in data_path]
    else:
        dataframe = pd.read_csv(data_path)
        report_dataset = str(data_path)
    target_column = resolve_target_column(dataframe)
    features = resolve_features(dataframe)
    required_columns = [*features, target_column]
    missing_columns = [column for column in required_columns if column not in dataframe]
    if missing_columns:
        raise ValueError(
            f"Dataset {data_path} is missing required columns: {', '.join(missing_columns)}"
        )
    X = dataframe[features]
    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts[0] / class_counts[1])
    configurations = [
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 400},
        {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 250},
        {"max_depth": 7, "learning_rate": 0.08, "n_estimators": 350},
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 200},
    ][:MAX_TRIALS]

    best_model = None
    best_metrics = None
    best_config = None
    trials = []

    for trial_number, config in enumerate(configurations, start=1):
        model = XGBClassifier(
            **config,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        trial = {"trial": trial_number, "config": config, **metrics}
        trials.append(trial)
        print(
            f"Trial {trial_number}/{len(configurations)} | "
            f"accuracy={metrics['accuracy']:.3f} | ROC-AUC={metrics['roc_auc']:.3f} | "
            f"PR-AUC={metrics['pr_auc']:.3f} | failure recall={metrics['failure_recall']:.3f}"
        )

        if best_metrics is None or (
            metrics["pr_auc"], metrics["roc_auc"], metrics["failure_recall"]
        ) > (
            best_metrics["pr_auc"], best_metrics["roc_auc"], best_metrics["failure_recall"]
        ):
            best_model, best_metrics, best_config = model, metrics, config

        if is_accepted(metrics):
            print("Acceptance targets met; continuing remaining trials.")

    joblib.dump(best_model, MODEL_PATH)
    created_at = datetime.now(timezone.utc).isoformat()
    if isinstance(report_dataset, list):
        dataset_fingerprint = [ModelRegistry.fingerprint(path) for path in data_path]
    else:
        dataset_fingerprint = ModelRegistry.fingerprint(data_path)
    report = {
        "created_at": created_at,
        "model_type": "XGBClassifier",
        "automl_trials": len(trials),
        "dataset": report_dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_name": dataset if dataset in [*DATASETS, "demo"] else "custom",
        "features": features,
        "target_column": target_column,
        "rows": int(len(dataframe)),
        "failure_rate": float(y.mean()),
        "targets": {"roc_auc": TARGET_ROC_AUC, "failure_recall": TARGET_RECALL},
        "status": "accepted" if is_accepted(best_metrics) else "needs_more_features",
        "best_config": best_config,
        "best_metrics": best_metrics,
        "trials": trials,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    registry = ModelRegistry(REGISTRY_DIR)
    run_id = registry.create_run_id()
    run_dir = registry.save_run(run_id, MODEL_PATH, report)
    report["run_id"] = run_id
    report["run_artifact_dir"] = str(run_dir)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions = (best_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    print("\nBest model:")
    print(classification_report(y_test, predictions))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved MLOps run: {run_id}")
    print(f"Training status: {report['status']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the predictive-maintenance model.")
    parser.add_argument(
        "--dataset",
        default="microsoft",
        help="Dataset name (microsoft, synthetic, or demo) or path to a CSV file.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available named datasets and exit.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        for name, path in DATASETS.items():
            print(f"{name}: {path}")
    else:
        train(args.dataset)
