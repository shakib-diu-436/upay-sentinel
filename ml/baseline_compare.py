from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
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


# ============================================================
# EVALUATION HELPER
# ============================================================

def evaluate_predictions(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict:

    report = classification_report(
        y_true,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# MAIN
# ============================================================

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
    # 1. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA)

    # --------------------------------------------------------
    # 2. Build the exact engineered features used by the
    #    trained 18-feature model.
    # --------------------------------------------------------

    df = build_features(df)

    target = "is_suspicious"

    # --------------------------------------------------------
    # 3. Same split used by training
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,
        random_state=42,
        stratify=df[target],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[target],
    )

    y_test = test_df[target]

    print("\n=== BASELINE COMPARISON ===")
    print(
        f"Validation: {len(val_df):,}"
    )
    print(
        f"Test      : {len(test_df):,}"
    )
    print(
        f"Suspicious in test: "
        f"{int(y_test.sum())}"
    )

    # --------------------------------------------------------
    # 4. Six transparent risk signals
    # --------------------------------------------------------

    six_signals = pd.DataFrame(
        {
            "new_recipient": (
                test_df["recipient_new"] == 1
            ),

            "device_changed": (
                test_df["device_changed"] == 1
            ),

            "location_changed": (
                test_df["location_changed"] == 1
            ),

            "high_amount": (
                test_df["amount_ratio"] >= 3.0
            ),

            "high_velocity": (
                test_df["transactions_last_1h"] >= 3
            ),

            "unusual_hour": (
                test_df["hour"] < 6
            ),
        },
        index=test_df.index,
    )

    signal_count = six_signals.sum(
        axis=1
    )

    # --------------------------------------------------------
    # 5. Rule baseline >= 2 signals
    # --------------------------------------------------------

    rule_2_predictions = (
        signal_count >= 2
    ).astype(int).to_numpy()

    rule_2 = evaluate_predictions(
        y_test,
        rule_2_predictions,
    )

    # --------------------------------------------------------
    # 6. Rule baseline >= 3 signals
    # --------------------------------------------------------

    rule_3_predictions = (
        signal_count >= 3
    ).astype(int).to_numpy()

    rule_3 = evaluate_predictions(
        y_test,
        rule_3_predictions,
    )

    # --------------------------------------------------------
    # 7. XGBoost
    # --------------------------------------------------------

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    features = package["features"]

    threshold = float(
        package["threshold"]
    )

    print("\n=== MODEL INFO ===")
    print(
        f"Model features: "
        f"{len(features)}"
    )

    print(
        f"Stored threshold: "
        f"{threshold:.4f}"
    )

    print("\nXGBoost features:")

    for feature in features:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Verify all model features exist
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features after feature engineering: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Predict XGBoost
    # --------------------------------------------------------

    test_probabilities = model.predict_proba(
        test_df[features]
    )[:, 1]

    xgb_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    xgb = evaluate_predictions(
        y_test,
        xgb_predictions,
    )

    xgb["roc_auc"] = roc_auc_score(
        y_test,
        test_probabilities,
    )

    xgb["pr_auc"] = average_precision_score(
        y_test,
        test_probabilities,
    )

    # --------------------------------------------------------
    # 8. Test-set comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST-SET COMPARISON")
    print("=" * 70)

    results = pd.DataFrame(
        [
            {
                "Method": "Rule >= 2 of 6",
                "Precision": rule_2[
                    "precision"
                ],
                "Recall": rule_2[
                    "recall"
                ],
                "F1": rule_2[
                    "f1"
                ],
                "ROC-AUC": np.nan,
            },

            {
                "Method": "Rule >= 3 of 6",
                "Precision": rule_3[
                    "precision"
                ],
                "Recall": rule_3[
                    "recall"
                ],
                "F1": rule_3[
                    "f1"
                ],
                "ROC-AUC": np.nan,
            },

            {
                "Method": "XGBoost",
                "Precision": xgb[
                    "precision"
                ],
                "Recall": xgb[
                    "recall"
                ],
                "F1": xgb[
                    "f1"
                ],
                "ROC-AUC": xgb[
                    "roc_auc"
                ],
            },
        ]
    )

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 9. Confusion matrices
    # --------------------------------------------------------

    print("\n=== CONFUSION MATRICES ===")

    print("\nRule >= 2 of 6:")
    print(
        np.array(
            [
                [
                    rule_2["tn"],
                    rule_2["fp"],
                ],
                [
                    rule_2["fn"],
                    rule_2["tp"],
                ],
            ]
        )
    )

    print("\nRule >= 3 of 6:")
    print(
        np.array(
            [
                [
                    rule_3["tn"],
                    rule_3["fp"],
                ],
                [
                    rule_3["fn"],
                    rule_3["tp"],
                ],
            ]
        )
    )

    print("\nXGBoost:")
    print(
        np.array(
            [
                [
                    xgb["tn"],
                    xgb["fp"],
                ],
                [
                    xgb["fn"],
                    xgb["tp"],
                ],
            ]
        )
    )

    # --------------------------------------------------------
    # 10. XGBoost vs 2-signal rule
    # --------------------------------------------------------

    print(
        "\n=== XGBOOST VS 2-SIGNAL RULE ==="
    )

    print(
        f"Precision change : "
        f"{xgb['precision'] - rule_2['precision']:+.4f}"
    )

    print(
        f"Recall change    : "
        f"{xgb['recall'] - rule_2['recall']:+.4f}"
    )

    print(
        f"F1 change        : "
        f"{xgb['f1'] - rule_2['f1']:+.4f}"
    )

    print(
        f"ROC-AUC          : "
        f"{xgb['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC           : "
        f"{xgb['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()