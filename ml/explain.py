from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import shap


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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python ml/train.py` first."
        )

    package = joblib.load(MODEL_PATH)

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
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python ml/generator.py` first."
        )

    return pd.read_csv(DATA_PATH)


def get_transaction(
    df: pd.DataFrame,
) -> pd.Series:

    # Usage:
    # python ml/explain.py TX001758

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


def main() -> None:

    print("\n" + "=" * 65)
    print("                 UPAY SENTINEL")
    print("              Explainable AI Engine")
    print("=" * 65)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    model, features, threshold = load_model()
    df = load_data()
    transaction = get_transaction(df)

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

    risk_score = probability * 100
    prediction = int(
        probability >= threshold
    )

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # XGBoost binary classification normally gives
    # one value per feature.
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    explanation = pd.DataFrame(
        {
            "feature": features,
            "value": [
                float(X.iloc[0][feature])
                for feature in features
            ],
            "shap_value": values,
        }
    )

    explanation["abs_shap"] = (
        explanation["shap_value"]
        .abs()
    )

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False,
    )

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    print("\nTRANSACTION")
    print("-" * 65)

    print(
        f"Transaction ID : "
        f"{transaction['transaction_id']}"
    )

    print(
        f"Customer ID    : "
        f"{transaction['customer_id']}"
    )

    print(
        f"Amount         : "
        f"৳{float(transaction['amount']):,.2f}"
    )

    print(
        f"Time           : "
        f"{transaction['timestamp']}"
    )

    print("\nAI ASSESSMENT")
    print("-" * 65)

    print(
        f"Risk Score     : "
        f"{risk_score:.2f} / 100"
    )

    print(
        f"Threshold      : "
        f"{threshold * 100:.2f}"
    )

    print(
        f"Assessment     : "
        f"{'HIGH RISK' if prediction else 'LOWER RISK'}"
    )

    # ---------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------

    print("\nWHY IS THIS TRANSACTION RISKY?")
    print("-" * 65)

    for index, row in explanation.head(8).iterrows():

        direction = (
            "↑ increases risk"
            if row["shap_value"] > 0
            else "↓ decreases risk"
        )

        print(
            f"{index + 1}. "
            f"{row['feature']:<30} "
            f"{row['shap_value']:+.4f} "
            f"{direction}"
        )

    # ---------------------------------------------------------
    # Human-readable feature explanation
    # ---------------------------------------------------------

    print("\nKEY RISK SIGNALS")
    print("-" * 65)

    if int(transaction["recipient_new"]) == 1:
        print("• New recipient detected")

    if int(transaction["device_changed"]) == 1:
        print("• Device changed from normal pattern")

    if int(transaction["location_changed"]) == 1:
        print("• Location changed from normal pattern")

    if float(transaction["amount_ratio"]) >= 3:
        print(
            f"• Transaction amount is "
            f"{float(transaction['amount_ratio']):.2f}× "
            f"the customer's typical amount"
        )

    if int(transaction["transactions_last_1h"]) >= 3:
        print(
            f"• High transaction activity: "
            f"{int(transaction['transactions_last_1h'])} "
            f"transactions in the last hour"
        )

    if float(
        transaction["behavioral_deviation_score"]
    ) >= 0.5:
        print(
            "• Significant behavioral deviation detected"
        )

    # ---------------------------------------------------------
    # Synthetic ground truth
    # ---------------------------------------------------------

    if "is_suspicious" in transaction.index:

        actual = int(
            transaction["is_suspicious"]
        )

        print("\nSYNTHETIC DATA CHECK")
        print("-" * 65)

        print(
            f"Actual Label : "
            f"{'SUSPICIOUS' if actual else 'NORMAL'}"
        )

        print(
            f"Prediction    : "
            f"{'SUSPICIOUS' if prediction else 'NORMAL'}"
        )

        print(
            f"Match         : "
            f"{'YES' if actual == prediction else 'NO'}"
        )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()