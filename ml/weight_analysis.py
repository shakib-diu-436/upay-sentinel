from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from train import (
    ENHANCED_FEATURES,
    build_features,
    find_best_threshold,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "transactions.csv"


WEIGHTS = [
    4.0,
    5.0,
    6.0,
    7.0,
    8.2879,
    9.0,
    10.0,
    12.0,
]


def evaluate_weight(
    weight: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=weight,
        random_state=42,
        n_jobs=-1,

        n_estimators=1200,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=1,

        subsample=0.90,
        colsample_bytree=0.95,

        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,

        early_stopping_rounds=50,
    )

    model.fit(
        X_train[ENHANCED_FEATURES],
        y_train,
        eval_set=[
            (
                X_val[ENHANCED_FEATURES],
                y_val,
            )
        ],
        verbose=False,
    )

    probabilities = model.predict_proba(
        X_val[ENHANCED_FEATURES]
    )[:, 1]

    threshold, validation_f1 = find_best_threshold(
        y_val,
        probabilities,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    report = classification_report(
        y_val,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "weight": weight,
        "model": model,
        "threshold": threshold,
        "f1": validation_f1,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "best_iteration": model.best_iteration,
    }


def main() -> None:

    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset not found. "
            "Run `python ml/generator.py` first."
        )

    df = pd.read_csv(DATA)

    df = build_features(df)

    X = df.drop(
        columns=["is_suspicious"]
    )

    y = df["is_suspicious"]

    # Same split as the main training pipeline.
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

    print("\n=== CLASS-WEIGHT SEARCH ===")

    results = []

    for weight in WEIGHTS:

        print(
            f"\nTesting scale_pos_weight = "
            f"{weight}"
        )

        result = evaluate_weight(
            weight=weight,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        results.append(result)

        print(
            f"Validation F1 : "
            f"{result['f1']:.4f}"
        )

        print(
            f"Precision      : "
            f"{result['precision']:.4f}"
        )

        print(
            f"Recall         : "
            f"{result['recall']:.4f}"
        )

        print(
            f"Threshold      : "
            f"{result['threshold']:.4f}"
        )

    # --------------------------------------------------------
    # Best weight from validation ONLY
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda result: result["f1"],
    )

    print("\n" + "=" * 70)
    print("BEST CLASS WEIGHT")
    print("=" * 70)

    print(
        f"Weight     : {best['weight']:.4f}"
    )

    print(
        f"Val F1     : {best['f1']:.4f}"
    )

    print(
        f"Val Precision : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Val Recall : "
        f"{best['recall']:.4f}"
    )

    print(
        f"Threshold  : "
        f"{best['threshold']:.4f}"
    )

    print(
        f"Best iteration : "
        f"{best['best_iteration']}"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    model = best["model"]

    test_probabilities = model.predict_proba(
        X_test[ENHANCED_FEATURES]
    )[:, 1]

    test_predictions = (
        test_probabilities >= best["threshold"]
    ).astype(int)

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
    ).ravel()

    print("\n" + "=" * 70)
    print("UNTOUCHED TEST RESULTS")
    print("=" * 70)

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"FPR       : "
        f"{fp / max(fp + tn, 1):.4f}"
    )

    print(
        f"FNR       : "
        f"{fn / max(fn + tp, 1):.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        np.array(
            [
                [tn, fp],
                [fn, tp],
            ]
        )
    )


if __name__ == "__main__":
    main()