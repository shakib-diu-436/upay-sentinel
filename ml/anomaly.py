from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


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
    / "anomaly_model.joblib"
)


FEATURES = [
    "amount_ratio",
    "hour_distance_from_usual",
    "transactions_last_1h",
    "transactions_last_24h",
    "recipient_new",
    "device_changed",
    "location_changed",
    "behavioral_deviation_score",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python ml/generator.py` first."
        )

    return pd.read_csv(DATA_PATH)


def get_transaction(
    df: pd.DataFrame,
) -> pd.Series:

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


def train_anomaly_model(
    df: pd.DataFrame,
) -> IsolationForest:

    X = df[FEATURES]

    model = IsolationForest(
        n_estimators=300,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X)

    return model


def main() -> None:

    print("\n" + "=" * 65)
    print("                 UPAY SENTINEL")
    print("          Behavioral Anomaly Engine")
    print("=" * 65)

    df = load_data()

    # ---------------------------------------------------------
    # Train anomaly detector
    # ---------------------------------------------------------

    model = train_anomaly_model(df)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Select transaction
    # ---------------------------------------------------------

    transaction = get_transaction(df)

    X = pd.DataFrame(
        [
            {
                feature: transaction[feature]
                for feature in FEATURES
            }
        ]
    )

    # ---------------------------------------------------------
    # Isolation Forest prediction
    # ---------------------------------------------------------

    prediction = int(
        model.predict(X)[0]
    )

    raw_score = float(
        model.decision_function(X)[0]
    )

    # Convert the raw score into a simple display score.
    anomaly_score = max(
        0.0,
        min(
            1.0,
            0.5 - raw_score,
        ),
    )

    anomaly_score_100 = anomaly_score * 100

    is_anomaly = prediction == -1

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

    print("\nBEHAVIOR ANALYSIS")
    print("-" * 65)

    print(
        f"Anomaly Score  : "
        f"{anomaly_score_100:.2f} / 100"
    )

    print(
        f"Assessment     : "
        f"{'ANOMALOUS' if is_anomaly else 'NORMAL BEHAVIOR'}"
    )

    print("\nBEHAVIOR SIGNALS")
    print("-" * 65)

    print(
        f"Amount Ratio             : "
        f"{float(transaction['amount_ratio']):.2f}x"
    )

    print(
        f"Hour Distance             : "
        f"{float(transaction['hour_distance_from_usual']):.0f}"
    )

    print(
        f"Transactions in 1h       : "
        f"{int(transaction['transactions_last_1h'])}"
    )

    print(
        f"Transactions in 24h      : "
        f"{int(transaction['transactions_last_24h'])}"
    )

    print(
        f"New Recipient             : "
        f"{'YES' if int(transaction['recipient_new']) else 'NO'}"
    )

    print(
        f"Device Changed            : "
        f"{'YES' if int(transaction['device_changed']) else 'NO'}"
    )

    print(
        f"Location Changed          : "
        f"{'YES' if int(transaction['location_changed']) else 'NO'}"
    )

    print(
        f"Behavioral Deviation      : "
        f"{float(transaction['behavioral_deviation_score']):.2f}"
    )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()