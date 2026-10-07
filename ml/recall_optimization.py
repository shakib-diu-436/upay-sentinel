from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from train import build_features


ROOT = Path(__file__).resolve().parents[1]

DATA = (
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


def evaluate(
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

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    f2 = (
        5 * precision * recall
        / max(
            4 * precision + recall,
            1e-12,
        )
    )

    fpr = (
        fp / max(fp + tn, 1)
    )

    fnr = (
        fn / max(fn + tp, 1)
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA)

    # Build exactly the same engineered features
    # used by the current 18-feature model.
    df = build_features(df)

    target = "is_suspicious"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # --------------------------------------------------------
    # Same split as training
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.40,
            random_state=42,
            stratify=y,
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp,
        )
    )

    # --------------------------------------------------------
    # Load current trained model
    # --------------------------------------------------------

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    features = package["features"]

    stored_threshold = float(
        package["threshold"]
    )

    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val[features]
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test[features]
    )[:, 1]

    print("\n=== CURRENT MODEL ===")
    print(
        f"Stored threshold : "
        f"{stored_threshold:.4f}"
    )

    print(
        f"Validation rows  : "
        f"{len(X_val):,}"
    )

    print(
        f"Test rows        : "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # Wide threshold grid
    # --------------------------------------------------------

    thresholds = np.arange(
        0.40,
        0.901,
        0.025,
    )

    validation_results = []

    for threshold in thresholds:

        result = evaluate(
            y_val,
            val_probabilities,
            float(threshold),
        )

        validation_results.append(
            result
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    print(
        "\n=== VALIDATION THRESHOLDS ==="
    )

    print(
        validation_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "f2",
                "fpr",
                "fnr",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_f1 = validation_df.loc[
        validation_df["f1"].idxmax()
    ]

    print(
        "\n=== BEST VALIDATION F1 ==="
    )

    print(
        f"Threshold : "
        f"{best_f1['threshold']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Best F2
    #
    # F2 gives more importance to recall than precision.
    # --------------------------------------------------------

    best_f2 = validation_df.loc[
        validation_df["f2"].idxmax()
    ]

    print(
        "\n=== BEST VALIDATION F2 ==="
    )

    print(
        f"Threshold : "
        f"{best_f2['threshold']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_f2['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f2['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f2['f1']:.4f}"
    )

    print(
        f"F2        : "
        f"{best_f2['f2']:.4f}"
    )

    # --------------------------------------------------------
    # Recall >= 60%
    #
    # Among thresholds achieving at least 60% recall,
    # select the one with the highest F1.
    # --------------------------------------------------------

    recall_60 = validation_df[
        validation_df["recall"] >= 0.60
    ].copy()

    print(
        "\n=== RECALL >= 60% ==="
    )

    if recall_60.empty:

        print(
            "No validation threshold "
            "reaches 60% recall."
        )

    else:

        recall_60 = recall_60.sort_values(
            by=[
                "f1",
                "precision",
            ],
            ascending=False,
        )

        print(
            recall_60[
                [
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "fpr",
                    "fnr",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # --------------------------------------------------------
    # Recall >= 55%
    # --------------------------------------------------------

    recall_55 = validation_df[
        validation_df["recall"] >= 0.55
    ].copy()

    print(
        "\n=== RECALL >= 55% ==="
    )

    if recall_55.empty:

        print(
            "No validation threshold "
            "reaches 55% recall."
        )

    else:

        recall_55 = recall_55.sort_values(
            by=[
                "f1",
                "precision",
            ],
            ascending=False,
        )

        print(
            recall_55[
                [
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "fpr",
                    "fnr",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )

    # --------------------------------------------------------
    # Select operating threshold
    #
    # Priority:
    #   1. Recall >= 60%
    #   2. Best F1 within that group
    #   3. Otherwise best F2
    # --------------------------------------------------------

    if not recall_60.empty:

        selected = recall_60.iloc[0]

        reason = (
            "Highest validation F1 among "
            "thresholds with recall >= 60%."
        )

    else:

        selected = best_f2

        reason = (
            "No threshold reached 60% recall; "
            "selected validation F2 optimum."
        )

    selected_threshold = float(
        selected["threshold"]
    )

    print(
        "\n=== SELECTED OPERATING POINT ==="
    )

    print(
        f"Threshold : "
        f"{selected_threshold:.4f}"
    )

    print(
        f"Precision : "
        f"{selected['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{selected['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{selected['f1']:.4f}"
    )

    print(
        f"F2        : "
        f"{selected['f2']:.4f}"
    )

    print(
        f"FPR       : "
        f"{selected['fpr']:.4f}"
    )

    print(
        f"FNR       : "
        f"{selected['fnr']:.4f}"
    )

    print(
        f"Reason    : "
        f"{reason}"
    )

    # --------------------------------------------------------
    # Final test evaluation
    #
    # The test set is untouched until this point.
    # --------------------------------------------------------

    test_result = evaluate(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    print(
        "\n=== UNTOUCHED TEST RESULT ==="
    )

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
        f"F2        : "
        f"{test_result['f2']:.4f}"
    )

    print(
        f"FPR       : "
        f"{test_result['fpr']:.4f}"
    )

    print(
        f"FNR       : "
        f"{test_result['fnr']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        np.array(
            [
                [
                    test_result["tn"],
                    test_result["fp"],
                ],
                [
                    test_result["fn"],
                    test_result["tp"],
                ],
            ]
        )
    )


if __name__ == "__main__":
    main()