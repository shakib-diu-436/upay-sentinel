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
        """
        Combine transaction risk and behavioral anomaly
        into a composite Sentinel risk severity score.

        Base contribution:
            70% transaction risk
            30% behavioral anomaly

        Corroboration bonus:
            Added when both signals are simultaneously high.

        IMPORTANT:
            This is a composite risk severity score,
            not a calibrated probability of fraud.
        """

        transaction = max(
            0.0,
            min(100.0, transaction_risk),
        ) / 100.0

        anomaly = max(
            0.0,
            min(100.0, anomaly_score),
        ) / 100.0

        # -----------------------------------------------------
        # Base contribution
        # -----------------------------------------------------

        base_risk = (
            0.70 * transaction
            + 0.30 * anomaly
        )

        # -----------------------------------------------------
        # Multi-signal corroboration
        # -----------------------------------------------------

        corroboration_bonus = (
            0.15
            * transaction
            * anomaly
        )

        final_score = (
            base_risk
            + corroboration_bonus
        ) * 100.0

        return max(
            0.0,
            min(100.0, final_score),
        )

    @staticmethod
    def classify(
        final_score: float,
    ) -> str:
        """
        Final Sentinel risk levels:

        0-44.99   -> LOW RISK
        45-69.99  -> REVIEW
        70-100    -> HIGH RISK
        """

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
                "Review transaction context and "
                "supporting signals before proceeding."
            )

        return (
            "No elevated risk detected; continue "
            "according to normal transaction controls."
        )

    def analyze(
        self,
        data: dict,
    ) -> dict:
        """
        Run all Sentinel intelligence layers.

        Flow:
            XGBoost transaction risk
                ↓
            Model threshold decision
                ↓
            Isolation Forest anomaly
                ↓
            Risk fusion
                ↓
            Sentinel risk classification
                ↓
            SHAP explanation
        """

        # -----------------------------------------------------
        # Transaction Risk Model
        # -----------------------------------------------------

        transaction_risk = (
            self.models.predict_transaction_risk(
                data
            )
        )

        # -----------------------------------------------------
        # XGBoost model-level classification
        # -----------------------------------------------------
        #
        # transaction_risk is returned as probability * 100.
        # Convert it back to 0-1 before comparing it with
        # the stored model threshold.
        #
        # Example:
        # threshold = 0.7346
        # probability = 0.80
        # 0.80 >= 0.7346
        # => SUSPICIOUS
        #

        model_probability = (
            transaction_risk / 100.0
        )

        model_decision = (
            "SUSPICIOUS"
            if model_probability >= self.models.threshold
            else "NORMAL"
        )

        # -----------------------------------------------------
        # Behavioral Anomaly Model
        # -----------------------------------------------------

        anomaly_score = (
            self.models.predict_anomaly(
                data
            )
        )

        # -----------------------------------------------------
        # Risk Fusion
        # -----------------------------------------------------

        final_score = (
            self.calculate_final_risk(
                transaction_risk,
                anomaly_score,
            )
        )

        # -----------------------------------------------------
        # Final Sentinel Risk Classification
        # -----------------------------------------------------

        risk_level = self.classify(
            final_score
        )

        # -----------------------------------------------------
        # Recommended Action
        # -----------------------------------------------------

        action = self.recommended_action(
            risk_level
        )

        # -----------------------------------------------------
        # Explainability
        # -----------------------------------------------------

        reasons = self.models.explain(
            data
        )

        # -----------------------------------------------------
        # API Response
        # -----------------------------------------------------

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

            # Stored threshold from the trained model.
            "model_threshold": round(
                self.models.threshold * 100,
                2,
            ),

            # XGBoost-only classification.
            # Separate from final Sentinel risk level.
            "model_decision": model_decision,

            "reasons": reasons,
        }