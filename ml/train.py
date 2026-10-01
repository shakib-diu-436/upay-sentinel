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
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "transactions.csv"
ART = ROOT / "artifacts"


FEATURES = [
    "amount",
    "recipient_new",
    "device_changed",
    "location_changed",
    "transactions_last_1h",
    "transactions_last_24h",
    "avg_amount_30d",
    "usual_transaction_hour",
    "hour",
    "account_age_days",
]

TARGET = "is_suspicious"


def find_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """
    Find the classification threshold that gives the
    best F1-score on the validation set.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.nanargmax(f1_scores))

    return (
        float(thresholds[best_index]),
        float(f1_scores[best_index]),
    )


def main() -> None:

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python ml/generator.py` first."
        )

    df = pd.read_csv(DATA)

    X = df[FEATURES]
    y = df[TARGET]

    print("\n=== UPAY SENTINEL DATASET ===")
    print(f"Total samples: {len(df):,}")
    print("\nClass distribution:")
    print(y.value_counts())
    print("\nClass share:")
    print(y.value_counts(normalize=True).round(4))


    # ---------------------------------------------------------
    # 2. Train / Validation / Test split
    # ---------------------------------------------------------

    # 60% Training
    # 20% Validation
    # 20% Test

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


    print("\n=== DATA SPLIT ===")
    print(f"Training   : {len(X_train):,}")
    print(f"Validation : {len(X_val):,}")
    print(f"Test       : {len(X_test):,}")


    # ---------------------------------------------------------
    # 3. Class imbalance handling
    # ---------------------------------------------------------

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    scale_pos_weight = negative_count / max(positive_count, 1)

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.4f}"
    )


    # ---------------------------------------------------------
    # 4. XGBoost model
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )


    print("\nTraining XGBoost model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")


    # ---------------------------------------------------------
    # 5. Validation prediction
    # ---------------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]


    # ---------------------------------------------------------
    # 6. Find best threshold using validation data
    # ---------------------------------------------------------

    best_threshold, validation_f1 = find_best_threshold(
        y_val,
        val_probabilities,
    )

    print("\n=== THRESHOLD TUNING ===")
    print(
        f"Best threshold : "
        f"{best_threshold:.4f}"
    )
    print(
        f"Validation F1  : "
        f"{validation_f1:.4f}"
    )


    # ---------------------------------------------------------
    # 7. Final test evaluation
    # ---------------------------------------------------------

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)


    print("\n=== TEST RESULTS ===")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            digits=4,
            zero_division=0,
        )
    )


    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    print(
        f"ROC-AUC : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.4f}"
    )


    # ---------------------------------------------------------
    # 8. Confusion matrix
    # ---------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        test_predictions,
    )

    print("\n=== CONFUSION MATRIX ===")
    print(
        "[[True Negative, False Positive],"
    )
    print(
        " [False Negative, True Positive]]"
    )
    print(matrix)


    # ---------------------------------------------------------
    # 9. Save model
    # ---------------------------------------------------------

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = ART / "risk_model.joblib"

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": best_threshold,
            "scale_pos_weight": scale_pos_weight,
        },
        model_path,
    )

    print(
        f"\nSaved model → {model_path}"
    )


if __name__ == "__main__":
    main()