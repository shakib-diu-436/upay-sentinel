from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
    "amount_ratio",
    "hour_distance_from_usual",
    "behavioral_deviation_score",
]

TARGET = "is_suspicious"


def find_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float]:

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


def train_candidate(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> dict:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        early_stopping_rounds=40,
    )

    model.fit(
        X_train[FEATURES],
        y_train,
        eval_set=[(X_val[FEATURES], y_val)],
        verbose=False,
    )

    val_probabilities = model.predict_proba(
        X_val[FEATURES]
    )[:, 1]

    threshold, val_f1 = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    val_precision = (
        (val_predictions[y_val.to_numpy() == 1].sum())
        / max((val_predictions == 1).sum(), 1)
    )

    val_recall = (
        (val_predictions[y_val.to_numpy() == 1].sum())
        / max((y_val.to_numpy() == 1).sum(), 1)
    )

    return {
        "model": model,
        "params": params,
        "threshold": threshold,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "best_iteration": model.best_iteration,
    }


def main() -> None:

    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python ml/generator.py` first."
        )

    df = pd.read_csv(DATA)

    X = df[FEATURES]
    y = df[TARGET]

    print("\n=== DATASET ===")
    print(f"Total samples: {len(df):,}")
    print(y.value_counts())

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

    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count

    scale_pos_weight = (
        negative_count / max(positive_count, 1)
    )

    print("\n=== SPLIT ===")
    print(f"Train : {len(X_train):,}")
    print(f"Val   : {len(X_val):,}")
    print(f"Test  : {len(X_test):,}")
    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ---------------------------------------------------------
    # Candidate configurations
    # ---------------------------------------------------------

    candidates = [
        {
            "n_estimators": 700,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 700,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 700,
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 700,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 700,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.10,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 700,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.90,
            "colsample_bytree": 0.85,
            "gamma": 0.10,
            "reg_alpha": 0.10,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 900,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.10,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 900,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.10,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
    ]

    results = []

    print("\n=== HYPERPARAMETER SEARCH ===")

    for index, params in enumerate(candidates, start=1):

        print(
            f"\n[{index}/{len(candidates)}] "
            f"depth={params['max_depth']} "
            f"lr={params['learning_rate']} "
            f"min_child={params['min_child_weight']}"
        )

        result = train_candidate(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scale_pos_weight=scale_pos_weight,
        )

        results.append(result)

        print(
            f"Validation F1: "
            f"{result['val_f1']:.4f} | "
            f"Precision: "
            f"{result['val_precision']:.4f} | "
            f"Recall: "
            f"{result['val_recall']:.4f} | "
            f"Threshold: "
            f"{result['threshold']:.4f}"
        )

    # ---------------------------------------------------------
    # Select best by validation F1
    # ---------------------------------------------------------

    best = max(
        results,
        key=lambda x: x["val_f1"],
    )

    print("\n" + "=" * 70)
    print("BEST VALIDATION MODEL")
    print("=" * 70)

    print(f"Validation F1 : {best['val_f1']:.4f}")
    print(f"Precision     : {best['val_precision']:.4f}")
    print(f"Recall        : {best['val_recall']:.4f}")
    print(f"Threshold     : {best['threshold']:.4f}")
    print(f"Best iteration: {best['best_iteration']}")

    print("\nBest parameters:")
    for key, value in best["params"].items():
        print(f"  {key}: {value}")

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------

    model = best["model"]

    test_probabilities = model.predict_proba(
        X_test[FEATURES]
    )[:, 1]

    test_predictions = (
        test_probabilities >= best["threshold"]
    ).astype(int)

    report = classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            digits=4,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Save
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
            "threshold": best["threshold"],
            "scale_pos_weight": scale_pos_weight,
            "validation_f1": best["val_f1"],
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc,
        },
        model_path,
    )

    print(
        f"\nSaved model → {model_path}"
    )


if __name__ == "__main__":
    main()