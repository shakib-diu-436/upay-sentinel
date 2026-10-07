from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "transactions.csv"
MODEL_PATH = ROOT / "artifacts" / "risk_model.joblib"


def metrics_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    false_positive_rate = (
        fp / max(fp + tn, 1)
    )

    false_negative_rate = (
        fn / max(fn + tp, 1)
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:

    if not DATA.exists():
        raise FileNotFoundError(DATA)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    df = pd.read_csv(DATA)

    target = "is_suspicious"

    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    features = saved["features"]
    current_threshold = float(saved["threshold"])

    X = df[features]
    y = df[target]

    # Same split used by training
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    val_probabilities = model.predict_proba(
        X_val[features]
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test[features]
    )[:, 1]

    print("\n=== CURRENT MODEL ===")
    print(f"Features          : {len(features)}")
    print(f"Stored threshold  : {current_threshold:.4f}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples      : {len(X_test):,}")

    # ---------------------------------------------------------
    # Threshold grid
    # ---------------------------------------------------------

    thresholds = np.arange(
        0.50,
        0.801,
        0.025,
    )

    validation_results = []

    for threshold in thresholds:
        result = metrics_at_threshold(
            y_val,
            val_probabilities,
            float(threshold),
        )
        validation_results.append(result)

    validation_df = pd.DataFrame(
        validation_results
    )

    print("\n=== VALIDATION THRESHOLD ANALYSIS ===")

    print(
        validation_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "fpr",
                "fnr",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------
    # Best validation F1
    # ---------------------------------------------------------

    best_f1_row = validation_df.loc[
        validation_df["f1"].idxmax()
    ]

    print("\n=== BEST VALIDATION F1 ===")

    print(
        f"Threshold : "
        f"{best_f1_row['threshold']:.4f}"
    )
    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )
    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )
    print(
        f"F1        : "
        f"{best_f1_row['f1']:.4f}"
    )
    print(
        f"FPR       : "
        f"{best_f1_row['fpr']:.4f}"
    )
    print(
        f"FNR       : "
        f"{best_f1_row['fnr']:.4f}"
    )

    # ---------------------------------------------------------
    # Recall-priority candidates
    # ---------------------------------------------------------

    recall_priority = validation_df[
        validation_df["recall"] >= 0.60
    ].sort_values(
        by="f1",
        ascending=False,
    )

    print("\n=== RECALL >= 60% CANDIDATES ===")

    if recall_priority.empty:
        print("No threshold reaches 60% recall.")
    else:
        print(
            recall_priority[
                [
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fpr",
                    "fnr",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ---------------------------------------------------------
    # Select threshold ONLY from validation
    #
    # Priority:
    # 1. Recall >= 60%
    # 2. Highest F1 among those
    # 3. Otherwise use best validation F1
    # ---------------------------------------------------------

    if not recall_priority.empty:

        selected_row = recall_priority.iloc[0]

        selection_reason = (
            "Highest validation F1 among thresholds "
            "with at least 60% recall."
        )

    else:

        selected_row = best_f1_row

        selection_reason = (
            "No threshold reached 60% recall; "
            "using validation-F1 optimum."
        )

    selected_threshold = float(
        selected_row["threshold"]
    )

    print("\n=== SELECTED OPERATING THRESHOLD ===")
    print(
        f"Threshold : "
        f"{selected_threshold:.4f}"
    )
    print(
        f"Reason    : "
        f"{selection_reason}"
    )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------

    test_result = metrics_at_threshold(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    print("\n=== UNTOUCHED TEST RESULT ===")

    print(
        f"Threshold : "
        f"{test_result['threshold']:.4f}"
    )
    print(
        f"Precision : "
        f"{test_result['precision']:.4f}"
    )
    print(
        f"Recall    : "
        f"{test_result['recall']:.4f}"
    )
    print(
        f"F1        : "
        f"{test_result['f1']:.4f}"
    )
    print(
        f"FPR       : "
        f"{test_result['fpr']:.4f}"
    )
    print(
        f"FNR       : "
        f"{test_result['fnr']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        f"[[{test_result['tn']} "
        f"{test_result['fp']}]"
    )
    print(
        f" [{test_result['fn']} "
        f"{test_result['tp']}]]"
    )


if __name__ == "__main__":
    main()