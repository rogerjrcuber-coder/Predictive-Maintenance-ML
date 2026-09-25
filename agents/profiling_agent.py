from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class ProfilingReport:
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    failure_rate: float
    class_imbalance_ratio: float
    quality_score: float
    recommended_model: str
    potential_leakage: bool

    def to_dict(self):
        return asdict(self)


class ProfilingAgent:
    def profile(self, dataframe: pd.DataFrame, target_column="failure"):
        rows, columns = dataframe.shape
        missing_values = int(dataframe.isna().sum().sum())
        duplicate_rows = int(dataframe.duplicated().sum())

        failure_rate = 0.0
        imbalance_ratio = 1.0
        if target_column in dataframe and rows:
            counts = dataframe[target_column].value_counts()
            failure_rate = float(dataframe[target_column].mean())
            if len(counts) > 1:
                imbalance_ratio = float(counts.max() / counts.min())

        quality_score = 100.0
        quality_score -= min(30.0, missing_values / max(rows, 1) * 100)
        quality_score -= min(10.0, duplicate_rows / max(rows, 1) * 100)
        if imbalance_ratio >= 10:
            quality_score -= 10.0
        quality_score = round(max(0.0, quality_score), 2)

        return ProfilingReport(
            rows=rows,
            columns=columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            failure_rate=round(failure_rate, 6),
            class_imbalance_ratio=round(imbalance_ratio, 2),
            quality_score=quality_score,
            recommended_model="XGBoost" if rows >= 10000 else "Random Forest",
            potential_leakage=self._detect_leakage(dataframe),
        )

    @staticmethod
    def _detect_leakage(dataframe):
        target = "failure"
        suspicious_names = {"failure", "failed", "target", "label"}
        return any(
            column.lower() in suspicious_names
            for column in dataframe.columns
            if column.lower() != target
        )
