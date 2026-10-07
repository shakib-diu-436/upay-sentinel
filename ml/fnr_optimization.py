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
MODEL_PATH = ROOT / "artifacts" / "risk_model.joblib"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

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


def metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

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

    fnr = (
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": fnr,
        "fpr": (
            fp / (fp + tn)
            if (fp + tn)
            else 0.0
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    features = package["features"]

    X = df[features]
    y = df["is_suspicious"].astype(int).to_numpy()

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

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    rows = []

    for threshold in np.arange(
        0.20,
        0.651,
        0.01,
    ):
        rows.append(
            metrics(
                y_val,
                val_prob,
                float(round(threshold, 2)),
            )
        )

    table = pd.DataFrame(rows)

    print()
    print("=" * 72)
    print("       FALSE-NEGATIVE / RECALL OPTIMIZATION")
    print("=" * 72)

    print()
    print("=== VALIDATION: BEST RECALL ===")

    best_recall = table.sort_values(
        ["recall", "precision"],
        ascending=[False, False],
    ).iloc[0]

    print(
        f"Threshold : {best_recall.threshold:.2f}"
    )
    print(
        f"Precision : {best_recall.precision:.4f}"
    )
    print(
        f"Recall    : {best_recall.recall:.4f}"
    )
    print(
        f"F1        : {best_recall.f1:.4f}"
    )
    print(
        f"FNR       : {best_recall.fnr:.4f}"
    )

    print()
    print("=== VALIDATION: PRECISION >= 50% ===")

    precision_floor = table[
        table["precision"] >= 0.50
    ].sort_values(
        ["recall", "fnr"],
        ascending=[False, True],
    )

    if not precision_floor.empty:
        selected = precision_floor.iloc[0]

        print(
            f"Threshold : {selected.threshold:.2f}"
        )
        print(
            f"Precision : {selected.precision:.4f}"
        )
        print(
            f"Recall    : {selected.recall:.4f}"
        )
        print(
            f"F1        : {selected.f1:.4f}"
        )
        print(
            f"FNR       : {selected.fnr:.4f}"
        )
    else:
        selected = best_recall

    # --------------------------------------------------------
    # IMPORTANT:
    # Threshold is selected using validation only.
    # --------------------------------------------------------

    selected_threshold = float(
        selected.threshold
    )

    test_result = metrics(
        y_test,
        test_prob,
        selected_threshold,
    )

    print()
    print("=" * 72)
    print("              UNTOUCHED TEST RESULT")
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

    y_test_pred = (
        test_prob >= selected_threshold
    ).astype(int)

    print()
    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_test_pred,
        )
    )


if __name__ == "__main__":
    main()