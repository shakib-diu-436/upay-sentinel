from __future__ import annotations

from typing import Any


class InvestigationService:
    """
    Evidence-grounded investigation assistant.

    This service does not independently decide whether
    a transaction is fraudulent.

    It summarizes model outputs and observable risk
    signals to support human investigation.
    """

    FEATURE_LABELS = {
        "amount": "Transaction amount",
        "device_changed": "Device changed",
        "recipient_new": "New recipient",
        "location_changed": "Location changed",
        "hour": "Transaction hour",
        "transactions_last_1h": "Transaction velocity (1 hour)",
        "transactions_last_24h": "Transaction activity (24 hours)",
        "avg_amount_30d": "30-day average transaction amount",
        "usual_transaction_hour": "Usual transaction hour",
        "account_age_days": "Account age",
    }

    def build_case(
        self,
        transaction: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build an investigation case from structured
        transaction data and model evidence.
        """

        final_risk = float(
            analysis["final_risk_score"]
        )

        risk_level = str(
            analysis["risk_level"]
        )

        transaction_risk = float(
            analysis["transaction_risk"]
        )

        anomaly_score = float(
            analysis["anomaly_score"]
        )

        # -----------------------------------------------------
        # Evidence
        # -----------------------------------------------------

        evidence: list[dict[str, Any]] = []

        if int(transaction["recipient_new"]) == 1:
            evidence.append(
                {
                    "type": "recipient",
                    "severity": "HIGH",
                    "title": "New recipient detected",
                    "detail": (
                        "The recipient has not been observed "
                        "in the customer's known transaction pattern."
                    ),
                }
            )

        if int(transaction["device_changed"]) == 1:
            evidence.append(
                {
                    "type": "device",
                    "severity": "HIGH",
                    "title": "Device changed",
                    "detail": (
                        "The transaction was initiated from a "
                        "device different from the customer's "
                        "normal device pattern."
                    ),
                }
            )

        if int(transaction["location_changed"]) == 1:
            evidence.append(
                {
                    "type": "location",
                    "severity": "MEDIUM",
                    "title": "Location changed",
                    "detail": (
                        "The transaction location differs from "
                        "the customer's normal location pattern."
                    ),
                }
            )

        amount_ratio = float(
            transaction["amount_ratio"]
        )

        if amount_ratio >= 3:
            evidence.append(
                {
                    "type": "amount",
                    "severity": "HIGH",
                    "title": "Unusual transaction amount",
                    "detail": (
                        f"The transaction amount is "
                        f"{amount_ratio:.2f}× the customer's "
                        "typical transaction amount."
                    ),
                }
            )

        transactions_1h = int(
            transaction["transactions_last_1h"]
        )

        if transactions_1h >= 3:
            evidence.append(
                {
                    "type": "velocity",
                    "severity": "HIGH",
                    "title": "Elevated transaction velocity",
                    "detail": (
                        f"{transactions_1h} transactions "
                        "were observed within the last hour."
                    ),
                }
            )

        behavioral_deviation = float(
            transaction[
                "behavioral_deviation_score"
            ]
        )

        if behavioral_deviation >= 0.5:
            evidence.append(
                {
                    "type": "behavior",
                    "severity": "HIGH",
                    "title": "Significant behavioral deviation",
                    "detail": (
                        f"Behavioral deviation score is "
                        f"{behavioral_deviation:.2f}."
                    ),
                }
            )

        # -----------------------------------------------------
        # SHAP reasons
        # -----------------------------------------------------

        model_reasons: list[dict[str, Any]] = []

        for reason in analysis.get(
            "reasons",
            [],
        ):

            feature = reason.get(
                "feature",
                "unknown",
            )

            model_reasons.append(
                {
                    "feature": feature,
                    "label": self.FEATURE_LABELS.get(
                        feature,
                        feature,
                    ),
                    "shap_value": round(
                        float(
                            reason.get(
                                "shap_value",
                                0.0,
                            )
                        ),
                        4,
                    ),
                    "direction": reason.get(
                        "direction",
                        "unknown",
                    ),
                }
            )

        # -----------------------------------------------------
        # Investigation summary
        # -----------------------------------------------------

        summary = self._generate_summary(
            transaction=transaction,
            analysis=analysis,
            evidence=evidence,
        )

        # -----------------------------------------------------
        # Human action
        # -----------------------------------------------------

        human_action = self._suggest_human_action(
            risk_level
        )

        # -----------------------------------------------------
        # Case status
        # -----------------------------------------------------

        case_status = (
            "REQUIRES INVESTIGATION"
            if risk_level == "HIGH RISK"
            else (
                "REVIEW RECOMMENDED"
                if risk_level == "REVIEW"
                else "NO ELEVATED RISK"
            )
        )

        return {
    "case_status": case_status,

    "transaction_id": transaction.get(
        "transaction_id"
    ),

    "customer_id": transaction.get(
        "customer_id"
    ),

    "recipient_id": transaction.get(
        "recipient_id"
    ),

    "timestamp": transaction.get(
        "timestamp"
    ),

    "risk_level": risk_level,

            "final_risk_score": round(
                final_risk,
                2,
            ),

            "transaction_risk": round(
                transaction_risk,
                2,
            ),

            "anomaly_score": round(
                anomaly_score,
                2,
            ),

            "summary": summary,

            "evidence": evidence,

            "model_reasons": model_reasons,

            "recommended_human_action": human_action,

            "notice": (
                "This assistant summarizes model evidence "
                "for human investigation. It does not make "
                "an autonomous financial decision."
            ),
        }

    def _generate_summary(
        self,
        transaction: dict[str, Any],
        analysis: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Generate a deterministic evidence-grounded
        investigation narrative.

        No unsupported conclusion is added.
        """

        risk_level = str(
            analysis["risk_level"]
        )

        amount = float(
            transaction["amount"]
        )

        amount_ratio = float(
            transaction["amount_ratio"]
        )

        final_risk = float(
            analysis["final_risk_score"]
        )

        transaction_risk = float(
            analysis["transaction_risk"]
        )

        anomaly_score = float(
            analysis["anomaly_score"]
        )

        if risk_level == "HIGH RISK":

            opening = (
                f"The transaction carries a composite "
                f"Sentinel risk score of {final_risk:.2f}/100 "
                f"and is classified as HIGH RISK."
            )

        elif risk_level == "REVIEW":

            opening = (
                f"The transaction carries a composite "
                f"Sentinel risk score of {final_risk:.2f}/100 "
                f"and requires additional review."
            )

        else:

            opening = (
                f"The transaction carries a composite "
                f"Sentinel risk score of {final_risk:.2f}/100 "
                f"with no elevated risk detected."
            )

        # Build evidence descriptions.
        evidence_text = []

        for item in evidence[:5]:
            evidence_text.append(
                item["detail"]
            )

        if evidence_text:

            evidence_sentence = (
                " Key observations include "
                + "; ".join(
                    evidence_text
                )
                + "."
            )

        else:

            evidence_sentence = (
                " No major predefined risk signals "
                "were observed."
            )

        model_sentence = (
            f" The transaction-risk model produced "
            f"{transaction_risk:.2f}/100, while the behavioral "
            f"anomaly engine produced {anomaly_score:.2f}/100."
        )

        amount_sentence = (
            f" The transaction amount is "
            f"৳{amount:,.2f}"
            f" ({amount_ratio:.2f}× the customer's typical amount)."
        )

        closing = (
            " These findings should be reviewed by an "
            "authorized human operator before any "
            "high-impact action is taken."
        )

        return (
            opening
            + evidence_sentence
            + model_sentence
            + amount_sentence
            + closing
        )

    @staticmethod
    def _suggest_human_action(
        risk_level: str,
    ) -> str:

        if risk_level == "HIGH RISK":

            return (
                "Review the transaction evidence, verify "
                "relevant customer/context signals, and "
                "escalate according to applicable risk controls."
            )

        if risk_level == "REVIEW":

            return (
                "Review the transaction context and supporting "
                "signals before taking further action."
            )

        return (
            "No elevated risk detected. Continue under "
            "normal transaction monitoring controls."
        )