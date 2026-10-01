from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[3]

RISK_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "risk_model.joblib"
)

ANOMALY_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "anomaly_model.joblib"
)


class ModelService:

    def __init__(self) -> None:

        if not RISK_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Risk model not found: {RISK_MODEL_PATH}"
            )

        if not ANOMALY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Anomaly model not found: {ANOMALY_MODEL_PATH}"
            )

        risk_package = joblib.load(
            RISK_MODEL_PATH
        )

        anomaly_package = joblib.load(
            ANOMALY_MODEL_PATH
        )

        self.risk_model = risk_package["model"]
        self.risk_features = risk_package["features"]
        self.threshold = float(
            risk_package["threshold"]
        )

        self.anomaly_model = anomaly_package["model"]
        self.anomaly_features = anomaly_package["features"]

        self.shap_explainer = shap.TreeExplainer(
            self.risk_model
        )

    def _risk_dataframe(
        self,
        data: dict,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                {
                    feature: data[feature]
                    for feature in self.risk_features
                }
            ]
        )

    def _anomaly_dataframe(
        self,
        data: dict,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                {
                    feature: data[feature]
                    for feature in self.anomaly_features
                }
            ]
        )

    def predict_transaction_risk(
        self,
        data: dict,
    ) -> float:

        X = self._risk_dataframe(data)

        probability = float(
            self.risk_model.predict_proba(X)[0][1]
        )

        return probability * 100.0

    def predict_anomaly(
        self,
        data: dict,
    ) -> float:

        X = self._anomaly_dataframe(data)

        raw_score = float(
            self.anomaly_model.decision_function(X)[0]
        )

        anomaly_score = max(
            0.0,
            min(
                1.0,
                0.5 - raw_score,
            ),
        )

        return anomaly_score * 100.0

    def explain(
        self,
        data: dict,
        top_n: int = 5,
    ) -> list[dict]:

        X = self._risk_dataframe(data)

        shap_values = self.shap_explainer.shap_values(X)

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        explanation = []

        for feature, value in zip(
            self.risk_features,
            values,
        ):
            shap_value = float(value)

            explanation.append(
                {
                    "feature": feature,
                    "shap_value": shap_value,
                    "direction": (
                        "increases risk"
                        if shap_value > 0
                        else "decreases risk"
                    ),
                }
            )

        explanation.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        return explanation[:top_n]