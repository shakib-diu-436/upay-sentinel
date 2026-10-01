from __future__ import annotations

from .model_service import ModelService


class RiskService:

    def __init__(self) -> None:
        self.models = ModelService()

    @staticmethod
    def calculate_final_risk(
        transaction_risk: float,
        anomaly_score: float,
    ) -> float:

        final_score = (
            0.70 * transaction_risk
            + 0.30 * anomaly_score
        )

        return max(
            0.0,
            min(
                100.0,
                final_score,
            ),
        )

    @staticmethod
    def classify(
        final_score: float,
    ) -> str:

        if final_score >= 70:
            return "HIGH RISK"

        if final_score >= 45:
            return "REVIEW"

        return "LOW RISK"

    @staticmethod
    def recommended_action(
        risk_level: str,
    ) -> str:

        if risk_level == "HIGH RISK":
            return (
                "Manual review and additional "
                "verification recommended."
            )

        if risk_level == "REVIEW":
            return (
                "Review transaction context "
                "before proceeding."
            )

        return (
            "No elevated risk detected; "
            "continue according to normal "
            "transaction controls."
        )

    def analyze(
        self,
        data: dict,
    ) -> dict:

        transaction_risk = (
            self.models.predict_transaction_risk(
                data
            )
        )

        anomaly_score = (
            self.models.predict_anomaly(
                data
            )
        )

        final_score = (
            self.calculate_final_risk(
                transaction_risk,
                anomaly_score,
            )
        )

        risk_level = self.classify(
            final_score
        )

        action = self.recommended_action(
            risk_level
        )

        reasons = self.models.explain(
            data
        )

        return {
            "transaction_risk": round(
                transaction_risk,
                2,
            ),
            "anomaly_score": round(
                anomaly_score,
                2,
            ),
            "final_risk_score": round(
                final_score,
                2,
            ),
            "risk_level": risk_level,
            "recommended_action": action,
            "model_threshold": round(
                self.models.threshold * 100,
                2,
            ),
            "reasons": reasons,
        }