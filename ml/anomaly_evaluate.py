from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
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

CURRENT_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "anomaly_model.joblib"
)

EVALUATED_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "anomaly_model_holdout.joblib"
)


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
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run `python ml/generator.py` first."
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# ENSURE DERIVED FEATURES
# ============================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # --------------------------------------------------------
    # Amount ratio
    # --------------------------------------------------------

    if "amount_ratio" not in result.columns:
        result["amount_ratio"] = (
            result["amount"]
            / result["avg_amount_30d"].clip(lower=1.0)
        )

    # --------------------------------------------------------
    # Circular hour distance
    # --------------------------------------------------------

    if "hour_distance_from_usual" not in result.columns:
        hour_difference = (
            result["hour"]
            - result["usual_transaction_hour"]
        ).abs()

        result["hour_distance_from_usual"] = np.minimum(
            hour_difference,
            24 - hour_difference,
        )

    # --------------------------------------------------------
    # Behavioral deviation score
    # --------------------------------------------------------

    if "behavioral_deviation_score" not in result.columns:
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

    missing = [
        feature
        for feature in ANOMALY_FEATURES
        if feature not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing anomaly features: {missing}"
        )

    return result


# ============================================================
# METRIC HELPERS
# ============================================================

def false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )


def false_negative_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return (
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("        UPAY SENTINEL — ISOLATION FOREST HOLD-OUT")
    print("=" * 72)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()
    df = prepare_features(df)

    y = df["is_suspicious"].astype(int).to_numpy()

    X = df[ANOMALY_FEATURES].copy()

    print()
    print("Dataset")
    print("-" * 72)
    print(f"Total rows       : {len(df):,}")
    print(f"Suspicious       : {int(y.sum()):,}")
    print(f"Normal           : {int((y == 0).sum()):,}")

    # --------------------------------------------------------
    # 60 / 20 / 20 split
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    print()
    print("Split")
    print("-" * 72)
    print(f"Train rows       : {len(X_train):,}")
    print(f"Validation rows  : {len(X_val):,}")
    print(f"Test rows        : {len(X_test):,}")

    # --------------------------------------------------------
    # Train Isolation Forest
    # --------------------------------------------------------
    #
    # The labels are NOT used for training.
    #
    # contamination=0.10 is fixed before test evaluation.
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_raw = model.decision_function(X_val)
    val_anomaly_score = -val_raw

    val_pred = (
        model.predict(X_val) == -1
    ).astype(int)

    print()
    print("Validation")
    print("-" * 72)
    print(
        f"Precision        : "
        f"{precision_score(y_val, val_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall           : "
        f"{recall_score(y_val, val_pred, zero_division=0):.4f}"
    )
    print(
        f"F1               : "
        f"{f1_score(y_val, val_pred, zero_division=0):.4f}"
    )
    print(
        f"ROC-AUC          : "
        f"{roc_auc_score(y_val, val_anomaly_score):.4f}"
    )
    print(
        f"PR-AUC            : "
        f"{average_precision_score(y_val, val_anomaly_score):.4f}"
    )

    # --------------------------------------------------------
    # Unseen TEST
    # --------------------------------------------------------

    test_raw = model.decision_function(X_test)

    # Higher value = more anomalous
    test_anomaly_score = -test_raw

    test_pred = (
        model.predict(X_test) == -1
    ).astype(int)

    precision = precision_score(
        y_test,
        test_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_anomaly_score,
    )

    pr_auc = average_precision_score(
        y_test,
        test_anomaly_score,
    )

    fpr = false_positive_rate(
        y_test,
        test_pred,
    )

    fnr = false_negative_rate(
        y_test,
        test_pred,
    )

    cm = confusion_matrix(
        y_test,
        test_pred,
    )

    print()
    print("=" * 72)
    print("                    UNTOUCHED TEST")
    print("=" * 72)

    print(
        f"Accuracy         : {accuracy:.4f}"
    )
    print(
        f"Precision        : {precision:.4f}"
    )
    print(
        f"Recall           : {recall:.4f}"
    )
    print(
        f"F1               : {f1:.4f}"
    )
    print(
        f"ROC-AUC          : {roc_auc:.4f}"
    )
    print(
        f"PR-AUC           : {pr_auc:.4f}"
    )
    print(
        f"FPR              : {fpr:.4f}"
    )
    print(
        f"FNR              : {fnr:.4f}"
    )

    print()
    print(
        f"Flagged as anomaly: "
        f"{int(test_pred.sum()):,} / {len(test_pred):,} "
        f"({test_pred.mean() * 100:.2f}%)"
    )

    print()
    print("Confusion Matrix")
    print("-" * 72)
    print(cm)

    # --------------------------------------------------------
    # Save evaluated model separately
    # --------------------------------------------------------

    package = {
        "model": model,
        "features": ANOMALY_FEATURES,
        "contamination": 0.10,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_accuracy": float(accuracy),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_fpr": float(fpr),
        "test_fnr": float(fnr),
        "confusion_matrix": cm.tolist(),
        "evaluation": "60/20/20 stratified hold-out",
    }

    joblib.dump(
        package,
        EVALUATED_MODEL_PATH,
    )

    print()
    print("-" * 72)
    print(
        f"Evaluated model saved to:\n"
        f"{EVALUATED_MODEL_PATH}"
    )
    print("-" * 72)


if __name__ == "__main__":
    main()