from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "transactions.csv"
)

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


def load_transaction_data() -> pd.DataFrame:
    """Load synthetic transaction dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. "
            "Run `python ml/generator.py` first."
        )

    return pd.read_csv(DATA_PATH)


def load_risk_model():
    """Load supervised transaction risk model."""

    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Risk model not found. "
            "Run `python ml/train.py` first."
        )

    package = joblib.load(RISK_MODEL_PATH)

    if not isinstance(package, dict):
        raise ValueError(
            "Unexpected risk model format."
        )

    return (
        package["model"],
        package["features"],
        float(package["threshold"]),
    )


def load_anomaly_model():
    """Load behavioral anomaly model."""

    if not ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Anomaly model not found. "
            "Run `python ml/anomaly.py` first."
        )

    package = joblib.load(ANOMALY_MODEL_PATH)

    if not isinstance(package, dict):
        raise ValueError(
            "Unexpected anomaly model format."
        )

    return (
        package["model"],
        package["features"],
    )


def get_transaction(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Select transaction by ID.

    Usage:
        python ml/risk_engine.py TX001758

    Without ID, TX001758-like random selection is avoided;
    instead a deterministic sample is used.
    """

    if len(sys.argv) > 1:

        transaction_id = sys.argv[1]

        matches = df[
            df["transaction_id"].astype(str)
            == str(transaction_id)
        ]

        if matches.empty:
            raise ValueError(
                f"Transaction not found: {transaction_id}"
            )

        return matches.iloc[0]

    return df.sample(
        n=1,
        random_state=42,
    ).iloc[0]


def calculate_transaction_risk(
    model,
    features: list[str],
    transaction: pd.Series,
) -> float:
    """Return transaction risk as 0-100."""

    X = pd.DataFrame(
        [
            {
                feature: transaction[feature]
                for feature in features
            }
        ]
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    return probability * 100.0


def calculate_behavior_anomaly(
    model,
    features: list[str],
    transaction: pd.Series,
) -> float:
    """
    Convert Isolation Forest output to a simple
    0-100 anomaly display score.

    Important:
    This is an anomaly severity score, not a probability
    of fraud.
    """

    X = pd.DataFrame(
        [
            {
                feature: transaction[feature]
                for feature in features
            }
        ]
    )

    raw_score = float(
        model.decision_function(X)[0]
    )

    # Convert Isolation Forest's decision score
    # into a human-readable anomaly severity.
    anomaly_score = 0.5 - raw_score

    anomaly_score = max(
        0.0,
        min(
            1.0,
            anomaly_score,
        ),
    )

    return anomaly_score * 100.0


def calculate_fused_risk(
    transaction_risk: float,
    anomaly_score: float,
) -> float:
    """
    Combine the two AI signals.

    Current prototype weighting:
        70% transaction risk
        30% behavioral anomaly

    These weights are intentionally transparent
    and should later be validated/tuned using
    held-out data.
    """

    fused_score = (
        0.70 * transaction_risk
        + 0.30 * anomaly_score
    )

    return max(
        0.0,
        min(
            100.0,
            fused_score,
        ),
    )


def classify_risk(
    final_score: float,
) -> str:
    """Convert final score into Sentinel risk level."""

    if final_score >= 70:
        return "HIGH RISK"

    if final_score >= 45:
        return "REVIEW"

    return "LOW RISK"


def recommended_action(
    risk_level: str,
) -> str:
    """
    Recommend a human-controlled action.

    The prototype does not autonomously block
    or approve consequential financial activity.
    """

    if risk_level == "HIGH RISK":
        return (
            "Manual review and additional verification recommended."
        )

    if risk_level == "REVIEW":
        return (
            "Review transaction context before proceeding."
        )

    return (
        "No elevated risk detected. Proceed normally."
    )


def main() -> None:

    print("\n" + "=" * 70)
    print("                    UPAY SENTINEL")
    print("                 AI RISK FUSION ENGINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    df = load_transaction_data()

    (
        risk_model,
        risk_features,
        risk_threshold,
    ) = load_risk_model()

    (
        anomaly_model,
        anomaly_features,
    ) = load_anomaly_model()

    transaction = get_transaction(df)

    # ---------------------------------------------------------
    # AI #1 — Transaction Risk
    # ---------------------------------------------------------

    transaction_risk = calculate_transaction_risk(
        risk_model,
        risk_features,
        transaction,
    )

    # ---------------------------------------------------------
    # AI #2 — Behavioral Anomaly
    # ---------------------------------------------------------

    anomaly_score = calculate_behavior_anomaly(
        anomaly_model,
        anomaly_features,
        transaction,
    )

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------

    final_risk = calculate_fused_risk(
        transaction_risk,
        anomaly_score,
    )

    final_level = classify_risk(
        final_risk
    )

    action = recommended_action(
        final_level
    )

    # ---------------------------------------------------------
    # Display transaction
    # ---------------------------------------------------------

    print("\nTRANSACTION")
    print("-" * 70)

    print(
        f"Transaction ID : "
        f"{transaction['transaction_id']}"
    )

    print(
        f"Customer ID    : "
        f"{transaction['customer_id']}"
    )

    print(
        f"Recipient ID   : "
        f"{transaction['recipient_id']}"
    )

    print(
        f"Amount         : "
        f"৳{float(transaction['amount']):,.2f}"
    )

    print(
        f"Time           : "
        f"{transaction['timestamp']}"
    )

    # ---------------------------------------------------------
    # AI signal results
    # ---------------------------------------------------------

    print("\nAI SIGNALS")
    print("-" * 70)

    print(
        f"Transaction Risk Score : "
        f"{transaction_risk:.2f} / 100"
    )

    print(
        f"Behavior Anomaly Score : "
        f"{anomaly_score:.2f} / 100"
    )

    print(
        f"Model Threshold        : "
        f"{risk_threshold * 100:.2f}"
    )

    # ---------------------------------------------------------
    # Final fusion
    # ---------------------------------------------------------

    print("\nSENTINEL DECISION")
    print("-" * 70)

    print(
        f"Final Risk Score       : "
        f"{final_risk:.2f} / 100"
    )

    print(
        f"Risk Level             : "
        f"{final_level}"
    )

    print(
        f"Recommended Action     : "
        f"{action}"
    )

    # ---------------------------------------------------------
    # Explain supporting signals
    # ---------------------------------------------------------

    print("\nSUPPORTING SIGNALS")
    print("-" * 70)

    if int(transaction["recipient_new"]) == 1:
        print("• New recipient")

    if int(transaction["device_changed"]) == 1:
        print("• Device changed")

    if int(transaction["location_changed"]) == 1:
        print("• Location changed")

    if float(transaction["amount_ratio"]) >= 3:
        print(
            f"• Amount is "
            f"{float(transaction['amount_ratio']):.2f}× "
            f"the customer's typical amount"
        )

    if int(transaction["transactions_last_1h"]) >= 3:
        print(
            f"• High transaction velocity: "
            f"{int(transaction['transactions_last_1h'])} "
            f"transactions in the last hour"
        )

    if float(
        transaction["behavioral_deviation_score"]
    ) >= 0.5:
        print(
            "• Significant behavioral deviation"
        )

    # ---------------------------------------------------------
    # Synthetic ground truth
    # ---------------------------------------------------------

    if "is_suspicious" in transaction.index:

        actual = int(
            transaction["is_suspicious"]
        )

        print("\nSYNTHETIC DATA CHECK")
        print("-" * 70)

        print(
            f"Actual Label : "
            f"{'SUSPICIOUS' if actual else 'NORMAL'}"
        )

        print(
            f"Fusion Result: "
            f"{'SUSPICIOUS' if final_level == 'HIGH RISK' else 'NORMAL/REVIEW'}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()