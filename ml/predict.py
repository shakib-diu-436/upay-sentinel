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

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "risk_model.joblib"
)


def load_model():
    """Load trained Sentinel model package."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python ml/train.py` first."
        )

    package = joblib.load(MODEL_PATH)

    # Current train.py saves a dictionary.
    if not isinstance(package, dict):
        raise ValueError(
            "Unexpected model format."
        )

    return (
        package["model"],
        package["features"],
        float(package["threshold"]),
    )


def load_data() -> pd.DataFrame:
    """Load synthetic transaction data."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python ml/generator.py` first."
        )

    return pd.read_csv(DATA_PATH)


def get_transaction(
    df: pd.DataFrame,
) -> pd.Series:

    # Optional transaction ID:
    # python ml/predict.py TX001234
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

    # Otherwise pick one random transaction.
    return df.sample(
        n=1,
        random_state=42,
    ).iloc[0]


def risk_level(
    probability: float,
    threshold: float,
) -> str:
    """
    Convert model probability into a simple
    human-readable assessment.

    Threshold is learned during validation.
    """

    if probability >= threshold:
        return "HIGH RISK"

    # This middle band is for UI readability.
    # The actual model decision boundary remains
    # the tuned threshold.
    if probability >= threshold * 0.60:
        return "REVIEW"

    return "LOW RISK"


def main() -> None:

    print("\n" + "=" * 55)
    print("              UPAY SENTINEL")
    print("        AI Transaction Risk Engine")
    print("=" * 55)

    # ---------------------------------------------------------
    # Load model and data
    # ---------------------------------------------------------

    model, features, threshold = load_model()

    df = load_data()

    transaction = get_transaction(df)

    # ---------------------------------------------------------
    # Prepare model input
    # ---------------------------------------------------------

    X = pd.DataFrame(
        [
            {
                feature: transaction[feature]
                for feature in features
            }
        ]
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    probability = float(
        model.predict_proba(X)[0][1]
    )

    risk_score = probability * 100.0

    assessment = risk_level(
        probability,
        threshold,
    )

    prediction = int(
        probability >= threshold
    )

    # ---------------------------------------------------------
    # Display transaction information
    # ---------------------------------------------------------

    print("\nTRANSACTION")
    print("-" * 55)

    print(
        f"Transaction ID   : "
        f"{transaction['transaction_id']}"
    )

    print(
        f"Customer ID      : "
        f"{transaction['customer_id']}"
    )

    print(
        f"Recipient ID     : "
        f"{transaction['recipient_id']}"
    )

    print(
        f"Amount           : "
        f"৳{float(transaction['amount']):,.2f}"
    )

    print(
        f"Time             : "
        f"{transaction['timestamp']}"
    )

    print(
        f"New Recipient    : "
        f"{'YES' if int(transaction['recipient_new']) else 'NO'}"
    )

    print(
        f"Device Changed   : "
        f"{'YES' if int(transaction['device_changed']) else 'NO'}"
    )

    print(
        f"Location Changed : "
        f"{'YES' if int(transaction['location_changed']) else 'NO'}"
    )

    print(
        f"Transactions 1h  : "
        f"{int(transaction['transactions_last_1h'])}"
    )

    print(
        f"Transactions 24h : "
        f"{int(transaction['transactions_last_24h'])}"
    )

    print(
        f"Amount Ratio     : "
        f"{float(transaction['amount_ratio']):.2f}x"
    )

    # ---------------------------------------------------------
    # AI result
    # ---------------------------------------------------------

    print("\nAI ASSESSMENT")
    print("-" * 55)

    print(
        f"Risk Score       : "
        f"{risk_score:.2f} / 100"
    )

    print(
        f"Decision Threshold: "
        f"{threshold * 100:.2f}"
    )

    print(
        f"Assessment       : "
        f"{assessment}"
    )

    print(
        f"Model Prediction : "
        f"{'SUSPICIOUS' if prediction else 'NORMAL'}"
    )

    # ---------------------------------------------------------
    # Ground-truth label
    # IMPORTANT:
    # This is only available in our synthetic dataset.
    # In a real system, this would not be shown to the
    # end user because the true label is unknown at prediction.
    # ---------------------------------------------------------

    if "is_suspicious" in transaction.index:

        actual = int(
            transaction["is_suspicious"]
        )

        print("\nSYNTHETIC DATA CHECK")
        print("-" * 55)

        print(
            f"Actual Label     : "
            f"{'SUSPICIOUS' if actual else 'NORMAL'}"
        )

        print(
            f"Prediction Match : "
            f"{'YES' if actual == prediction else 'NO'}"
        )

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()