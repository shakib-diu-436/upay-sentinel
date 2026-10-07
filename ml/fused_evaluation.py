from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
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
    / "anomaly_model_holdout.joblib"
)


# ============================================================
# FUSION SETTINGS
# ============================================================

TRANSACTION_WEIGHT = 0.70
ANOMALY_WEIGHT = 0.30
INTERACTION_WEIGHT = 0.15


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Derived behavioural features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Enhanced XGBoost features
    # --------------------------------------------------------

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


# ============================================================
# FUSION
# ============================================================

def calculate_fused_score(
    transaction_risk: np.ndarray,
    anomaly_score: np.ndarray,
) -> np.ndarray:

    transaction_norm = (
        transaction_risk / 100.0
    )

    anomaly_norm = (
        anomaly_score / 100.0
    )

    base = (
        TRANSACTION_WEIGHT
        * transaction_norm
        + ANOMALY_WEIGHT
        * anomaly_norm
    )

    interaction = (
        INTERACTION_WEIGHT
        * transaction_norm
        * anomaly_norm
    )

    final = (
        base
        + interaction
    ) * 100.0

    return np.clip(
        final,
        0.0,
        100.0,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:

    y_pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )

    return {
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


# ============================================================
# TOP-K METRICS
# ============================================================

def top_k_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    percentage: float,
) -> dict:

    count = max(
        1,
        int(
            len(scores)
            * percentage
            / 100.0
        ),
    )

    ranking = np.argsort(
        scores
    )[::-1]

    selected = ranking[:count]

    y_selected = y_true[selected]

    precision = float(
        y_selected.mean()
    )

    captured = int(
        y_selected.sum()
    )

    total_positive = int(
        y_true.sum()
    )

    recall = (
        captured / total_positive
        if total_positive
        else 0.0
    )

    return {
        "alerts": count,
        "precision": precision,
        "recall": recall,
        "captured": captured,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("        UPAY SENTINEL — FUSED SCORE EVALUATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    y = df[
        "is_suspicious"
    ].astype(int).to_numpy()

    # --------------------------------------------------------
    # Same 60 / 20 / 20 split
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        df,
        y,
        test_size=0.40,
        stratify=y,
        random_state=42,
    )

    _, X_test, _, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    risk_package = joblib.load(
        RISK_MODEL_PATH
    )

    anomaly_package = joblib.load(
        ANOMALY_MODEL_PATH
    )

    risk_model = risk_package[
        "model"
    ]

    risk_features = risk_package[
        "features"
    ]

    anomaly_model = anomaly_package[
        "model"
    ]

    # --------------------------------------------------------
    # XGBoost probability
    # --------------------------------------------------------

    X_test_risk = X_test[
        risk_features
    ]

    transaction_probability = (
        risk_model
        .predict_proba(
            X_test_risk
        )[:, 1]
    )

    transaction_risk = (
        transaction_probability
        * 100.0
    )

    # --------------------------------------------------------
    # Isolation Forest anomaly
    # --------------------------------------------------------

    X_test_anomaly = X_test[
        ANOMALY_FEATURES
    ]

    raw_anomaly = (
        anomaly_model
        .decision_function(
            X_test_anomaly
        )
    )

    anomaly_score = (
        0.5
        - raw_anomaly
    )

    anomaly_score = np.clip(
        anomaly_score,
        0.0,
        1.0,
    ) * 100.0

    # --------------------------------------------------------
    # Fused Sentinel score
    # --------------------------------------------------------

    fused_score = calculate_fused_score(
        transaction_risk,
        anomaly_score,
    )

    print()
    print("Test rows:", len(y_test))

    print()
    print("Score Summary")
    print("-" * 72)
    print(
        f"Transaction risk mean : "
        f"{transaction_risk.mean():.2f}"
    )
    print(
        f"Anomaly score mean    : "
        f"{anomaly_score.mean():.2f}"
    )
    print(
        f"Fused score mean      : "
        f"{fused_score.mean():.2f}"
    )

    # --------------------------------------------------------
    # Fused ROC / PR
    # --------------------------------------------------------

    fused_roc_auc = roc_auc_score(
        y_test,
        fused_score,
    )

    fused_pr_auc = average_precision_score(
        y_test,
        fused_score,
    )

    print()
    print("Ranking Metrics")
    print("-" * 72)
    print(
        f"Fused ROC-AUC : "
        f"{fused_roc_auc:.4f}"
    )
    print(
        f"Fused PR-AUC  : "
        f"{fused_pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # HIGH RISK = 70+
    # --------------------------------------------------------

    high = calculate_metrics(
        y_test,
        fused_score,
        70.0,
    )

    print()
    print("=" * 72)
    print("             FUSED SCORE — HIGH RISK (>= 70)")
    print("=" * 72)

    print(
        f"Precision : {high['precision']:.4f}"
    )
    print(
        f"Recall    : {high['recall']:.4f}"
    )
    print(
        f"F1        : {high['f1']:.4f}"
    )
    print(
        f"FPR       : {high['fpr']:.4f}"
    )
    print(
        f"FNR       : {high['fnr']:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        np.array(
            [
                [
                    high["tn"],
                    high["fp"],
                ],
                [
                    high["fn"],
                    high["tp"],
                ],
            ]
        )
    )

    # --------------------------------------------------------
    # REVIEW+ = 45+
    # --------------------------------------------------------

    review = calculate_metrics(
        y_test,
        fused_score,
        45.0,
    )

    print()
    print("=" * 72)
    print("             FUSED SCORE — REVIEW+ (>= 45)")
    print("=" * 72)

    print(
        f"Precision : {review['precision']:.4f}"
    )
    print(
        f"Recall    : {review['recall']:.4f}"
    )
    print(
        f"F1        : {review['f1']:.4f}"
    )
    print(
        f"FPR       : {review['fpr']:.4f}"
    )
    print(
        f"FNR       : {review['fnr']:.4f}"
    )

    # --------------------------------------------------------
    # TOP-K
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("                    TOP-K ALERT ANALYSIS")
    print("=" * 72)

    for percentage in (
        5.0,
        10.0,
        20.0,
    ):

        result = top_k_metrics(
            y_test,
            fused_score,
            percentage,
        )

        print()
        print(
            f"Top {percentage:.0f}% alerts"
        )
        print(
            f"Alerts captured : "
            f"{result['alerts']:,}"
        )
        print(
            f"Suspicious found: "
            f"{result['captured']:,}"
        )
        print(
            f"Precision       : "
            f"{result['precision']:.4f}"
        )
        print(
            f"Recall          : "
            f"{result['recall']:.4f}"
        )

    print()
    print("=" * 72)
    print("                         DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()