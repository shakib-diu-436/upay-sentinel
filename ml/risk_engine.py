from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

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


# ============================================================
# DATA LOADING
# ============================================================

def load_transaction_data() -> pd.DataFrame:
    """Load the synthetic transaction dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. "
            "Run `python ml/generator.py` first."
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# MODEL LOADING
# ============================================================

def load_risk_model():
    """Load the trained supervised transaction risk model."""

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
    """Load the trained behavioral anomaly model."""

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


# ============================================================
# TRANSACTION SELECTION
# ============================================================

def get_transaction(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Select a transaction by ID.

    Usage:
        python ml/risk_engine.py TX001758

    Without a transaction ID, a deterministic sample is used.
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


# ============================================================
# TRANSACTION RISK
# ============================================================

def calculate_transaction_risk(
    model,
    features: list[str],
    transaction: pd.Series,
) -> float:
    """
    Calculate transaction risk score from 0 to 100.
    """

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


# ============================================================
# BEHAVIORAL ANOMALY
# ============================================================

def calculate_behavior_anomaly(
    model,
    features: list[str],
    transaction: pd.Series,
) -> float:
    """
    Convert Isolation Forest output into a 0-100
    behavioral anomaly severity score.

    IMPORTANT:
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

    # Convert Isolation Forest decision score to
    # a bounded human-readable severity score.
    anomaly_score = 0.5 - raw_score

    anomaly_score = max(
        0.0,
        min(
            1.0,
            anomaly_score,
        ),
    )

    return anomaly_score * 100.0


# ============================================================
# RISK FUSION
# ============================================================

def calculate_fused_risk(
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
        When BOTH signals are high, the agreement between
        the two models increases the final risk score.

    IMPORTANT:
        This is a composite risk severity score.
        It is NOT a calibrated probability of fraud.
    """

    # --------------------------------------------------------
    # Normalize both scores
    # --------------------------------------------------------

    transaction = max(
        0.0,
        min(100.0, transaction_risk),
    ) / 100.0

    anomaly = max(
        0.0,
        min(100.0, anomaly_score),
    ) / 100.0

    # --------------------------------------------------------
    # Base fusion
    # --------------------------------------------------------

    base_risk = (
        0.70 * transaction
        + 0.30 * anomaly
    )

    # --------------------------------------------------------
    # Multi-signal corroboration
    #
    # If both models agree that a transaction is risky,
    # the combined score receives an additional bounded
    # contribution.
    # --------------------------------------------------------

    corroboration_bonus = (
        0.15
        * transaction
        * anomaly
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    fused_score = (
        base_risk
        + corroboration_bonus
    ) * 100.0

    return max(
        0.0,
        min(
            100.0,
            fused_score,
        ),
    )


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk(
    final_score: float,
) -> str:
    """
    Convert final composite score into a Sentinel level.

        0 - 44.99   LOW RISK
        45 - 69.99  REVIEW
        70 - 100    HIGH RISK
    """

    if final_score >= 70:
        return "HIGH RISK"

    if final_score >= 45:
        return "REVIEW"

    return "LOW RISK"


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(
    risk_level: str,
) -> str:
    """
    Provide a human-controlled operational recommendation.

    The system does not autonomously block or approve
    consequential financial transactions.
    """

    if risk_level == "HIGH RISK":
        return (
            "Manual review and additional verification "
            "recommended."
        )

    if risk_level == "REVIEW":
        return (
            "Review transaction context and supporting "
            "signals before proceeding."
        )

    return (
        "No elevated risk detected; continue according "
        "to normal transaction controls."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("                    UPAY SENTINEL")
    print("                 AI RISK FUSION ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data and models
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # AI #1 — Transaction Risk
    # --------------------------------------------------------

    transaction_risk = calculate_transaction_risk(
        risk_model,
        risk_features,
        transaction,
    )

    # --------------------------------------------------------
    # AI #2 — Behavioral Anomaly
    # --------------------------------------------------------

    anomaly_score = calculate_behavior_anomaly(
        anomaly_model,
        anomaly_features,
        transaction,
    )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Transaction details
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # AI Signals
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final Sentinel Decision
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Supporting Signals
    # --------------------------------------------------------

    print("\nSUPPORTING SIGNALS")
    print("-" * 70)

    signal_count = 0

    if int(transaction["recipient_new"]) == 1:
        print("• New recipient")
        signal_count += 1

    if int(transaction["device_changed"]) == 1:
        print("• Device changed")
        signal_count += 1

    if int(transaction["location_changed"]) == 1:
        print("• Location changed")
        signal_count += 1

    if float(transaction["amount_ratio"]) >= 3:
        print(
            f"• Amount is "
            f"{float(transaction['amount_ratio']):.2f}× "
            f"the customer's typical amount"
        )
        signal_count += 1

    if int(transaction["transactions_last_1h"]) >= 3:
        print(
            f"• High transaction velocity: "
            f"{int(transaction['transactions_last_1h'])} "
            f"transactions in the last hour"
        )
        signal_count += 1

    if float(
        transaction["behavioral_deviation_score"]
    ) >= 0.5:
        print(
            "• Significant behavioral deviation"
        )
        signal_count += 1

    if signal_count == 0:
        print(
            "• No major predefined risk signals detected"
        )

    # --------------------------------------------------------
    # Synthetic Ground Truth
    #
    # This section exists ONLY for development/testing.
    # It must not be shown to end users in a real system.
    # --------------------------------------------------------

    if "is_suspicious" in transaction.index:

        actual = int(
            transaction["is_suspicious"]
        )

        predicted_suspicious = (
            final_level == "HIGH RISK"
        )

        print("\nSYNTHETIC DATA CHECK")
        print("-" * 70)

        print(
            f"Actual Label : "
            f"{'SUSPICIOUS' if actual else 'NORMAL'}"
        )

        print(
            f"Fusion Result: "
            f"{'SUSPICIOUS' if predicted_suspicious else final_level}"
        )

        print(
            f"Prediction Match: "
            f"{'YES' if actual == int(predicted_suspicious) else 'NO'}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()