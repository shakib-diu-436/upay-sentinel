from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "synthetic" / "transactions.csv"
RISK_MODEL_PATH = ROOT / "artifacts" / "risk_model.joblib"
ANOMALY_MODEL_PATH = (
    ROOT / "artifacts" / "anomaly_model_holdout.joblib"
)


ANOMALY_FEATURES = [
    "amount_ratio",
    "hour_distance_from_usual",
    "transactions_last_1h",
    "transactions_last_24h",
    "recipient_new",
    "device_changed",
    "location_changed",
    "behavioral_deviation_score",
]


TRANSACTION_WEIGHT = 0.70
ANOMALY_WEIGHT = 0.30
INTERACTION_WEIGHT = 0.15


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["amount_ratio"] = (
        result["amount"]
        / result["avg_amount_30d"].clip(lower=1.0)
    )

    hour_difference = (
        result["hour"]
        - result["usual_transaction_hour"]
    ).abs()

    result["hour_distance_from_usual"] = np.minimum(
        hour_difference,
        24 - hour_difference,
    )

    result["behavioral_deviation_score"] = np.clip(
        (
            0.40
            * np.minimum(
                result["amount_ratio"] / 8.0,
                2.0,
            )
            + 0.20
            * np.minimum(
                result["hour_distance_from_usual"] / 8.0,
                2.0,
            )
            + 0.15 * result["device_changed"]
            + 0.15 * result["location_changed"]
            + 0.10 * result["recipient_new"]
        ),
        0.0,
        1.0,
    )

    result["unusual_hour"] = (
        result["hour"] < 6
    ).astype(int)

    result["log_amount_ratio"] = np.log1p(
        result["amount_ratio"].clip(lower=0.0)
    )

    result["high_amount_flag"] = (
        result["amount_ratio"] >= 3.0
    ).astype(int)

    result["high_velocity_flag"] = (
        result["transactions_last_1h"] >= 3
    ).astype(int)

    result["risk_signal_count"] = (
        result["recipient_new"]
        + result["device_changed"]
        + result["location_changed"]
        + result["high_amount_flag"]
        + result["high_velocity_flag"]
        + result["unusual_hour"]
    ).astype(int)

    return result


def fused_score(
    transaction_risk: np.ndarray,
    anomaly_score: np.ndarray,
) -> np.ndarray:

    t = transaction_risk / 100.0
    a = anomaly_score / 100.0

    result = (
        TRANSACTION_WEIGHT * t
        + ANOMALY_WEIGHT * a
        + INTERACTION_WEIGHT * t * a
    ) * 100.0

    return np.clip(result, 0.0, 100.0)


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:

    prediction = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
    ).ravel()

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if fn + tp
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:

    print()
    print("=" * 72)
    print("      UPAY SENTINEL — FUSED FNR OPTIMIZATION")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    y = df["is_suspicious"].astype(int).to_numpy()

    train_df, temp_df, y_train, y_temp = train_test_split(
        df,
        y,
        test_size=0.40,
        stratify=y,
        random_state=42,
    )

    val_df, test_df, y_val, y_test = train_test_split(
        temp_df,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    risk_package = joblib.load(RISK_MODEL_PATH)
    anomaly_package = joblib.load(ANOMALY_MODEL_PATH)

    risk_model = risk_package["model"]
    risk_features = risk_package["features"]

    anomaly_model = anomaly_package["model"]

    # --------------------------------------------------------
    # Validation scores
    # --------------------------------------------------------

    val_transaction = (
        risk_model.predict_proba(
            val_df[risk_features]
        )[:, 1]
        * 100.0
    )

    val_raw_anomaly = anomaly_model.decision_function(
        val_df[ANOMALY_FEATURES]
    )

    val_anomaly = np.clip(
        0.5 - val_raw_anomaly,
        0.0,
        1.0,
    ) * 100.0

    val_fused = fused_score(
        val_transaction,
        val_anomaly,
    )

    # --------------------------------------------------------
    # Test scores
    # --------------------------------------------------------

    test_transaction = (
        risk_model.predict_proba(
            test_df[risk_features]
        )[:, 1]
        * 100.0
    )

    test_raw_anomaly = anomaly_model.decision_function(
        test_df[ANOMALY_FEATURES]
    )

    test_anomaly = np.clip(
        0.5 - test_raw_anomaly,
        0.0,
        1.0,
    ) * 100.0

    test_fused = fused_score(
        test_transaction,
        test_anomaly,
    )

    # --------------------------------------------------------
    # Threshold search — VALIDATION ONLY
    # --------------------------------------------------------

    rows = []

    for threshold in np.arange(
        30.0,
        75.01,
        1.0,
    ):
        result = evaluate(
            y_val,
            val_fused,
            float(threshold),
        )

        rows.append(result)

    table = pd.DataFrame(rows)

    print()
    print("=== VALIDATION: PRECISION >= 50% ===")

    eligible = table[
        table["precision"] >= 0.50
    ].sort_values(
        ["recall", "f1"],
        ascending=[False, False],
    )

    if eligible.empty:
        selected = table.sort_values(
            ["f1", "recall"],
            ascending=[False, False],
        ).iloc[0]
    else:
        selected = eligible.iloc[0]

    print()
    print(
        f"Threshold : {selected['threshold']:.2f}"
    )
    print(
        f"Precision : {selected['precision']:.4f}"
    )
    print(
        f"Recall    : {selected['recall']:.4f}"
    )
    print(
        f"F1        : {selected['f1']:.4f}"
    )
    print(
        f"FNR       : {selected['fnr']:.4f}"
    )
    print(
        f"FPR       : {selected['fpr']:.4f}"
    )

    selected_threshold = float(
        selected["threshold"]
    )

    # --------------------------------------------------------
    # Unseen TEST
    # --------------------------------------------------------

    test_result = evaluate(
        y_test,
        test_fused,
        selected_threshold,
    )

    print()
    print("=" * 72)
    print("                UNTOUCHED TEST RESULT")
    print("=" * 72)

    print(
        f"Threshold : {test_result['threshold']:.2f}"
    )
    print(
        f"Precision : {test_result['precision']:.4f}"
    )
    print(
        f"Recall    : {test_result['recall']:.4f}"
    )
    print(
        f"F1        : {test_result['f1']:.4f}"
    )
    print(
        f"FPR       : {test_result['fpr']:.4f}"
    )
    print(
        f"FNR       : {test_result['fnr']:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            (
                test_fused >= selected_threshold
            ).astype(int),
        )
    )


if __name__ == "__main__":
    main()