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


TARGET = "is_suspicious"


# ============================================================
# FEATURE SETS
# ============================================================

BASE_FEATURES = [
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


ENGINEERED_FEATURES = [
    "unusual_hour",
    "log_amount_ratio",
    "high_amount_flag",
    "high_velocity_flag",
    "risk_signal_count",
]


ENHANCED_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build additional interpretable risk features from
    already-available transaction and behavioral fields.

    These are derived from observable transaction context.
    No target label is used here.
    """

    result = df.copy()

    # --------------------------------------------------------
    # 1. Unusual transaction hour
    # --------------------------------------------------------
    # The synthetic generator defines the unusual period
    # as before 06:00.
    result["unusual_hour"] = (
        result["hour"] < 6
    ).astype(int)

    # --------------------------------------------------------
    # 2. Log amount ratio
    # --------------------------------------------------------
    result["log_amount_ratio"] = np.log1p(
        result["amount_ratio"].clip(
            lower=0.0,
        )
    )

    # --------------------------------------------------------
    # 3. High amount flag
    # --------------------------------------------------------
    result["high_amount_flag"] = (
        result["amount_ratio"] >= 3.0
    ).astype(int)

    # --------------------------------------------------------
    # 4. High velocity flag
    # --------------------------------------------------------
    result["high_velocity_flag"] = (
        result["transactions_last_1h"] >= 3
    ).astype(int)

    # --------------------------------------------------------
    # 5. Number of simultaneous risk signals
    # --------------------------------------------------------
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
# THRESHOLD TUNING
# ============================================================

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
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(
        np.nanargmax(f1_scores)
    )

    return (
        float(thresholds[best_index]),
        float(f1_scores[best_index]),
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    features: list[str],
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> dict:
    """
    Train one candidate and evaluate only on validation data.
    Test data is not used here.
    """

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
        X_train[features],
        y_train,
        eval_set=[
            (
                X_val[features],
                y_val,
            )
        ],
        verbose=False,
    )

    val_probabilities = model.predict_proba(
        X_val[features]
    )[:, 1]

    threshold, validation_f1 = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    validation_report = classification_report(
        y_val,
        val_predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": model,
        "features": features,
        "params": params,
        "threshold": threshold,
        "validation_f1": validation_f1,
        "validation_precision": (
            validation_report["1"]["precision"]
        ),
        "validation_recall": (
            validation_report["1"]["recall"]
        ),
        "best_iteration": model.best_iteration,
    }


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(
    result: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:

    model = result["model"]
    features = result["features"]
    threshold = result["threshold"]

    test_probabilities = model.predict_proba(
        X_test[features]
    )[:, 1]

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
    )

    result["test_precision"] = (
        report["1"]["precision"]
    )

    result["test_recall"] = (
        report["1"]["recall"]
    )

    result["test_f1"] = (
        report["1"]["f1-score"]
    )

    result["test_roc_auc"] = roc_auc_score(
        y_test,
        test_probabilities,
    )

    result["test_pr_auc"] = average_precision_score(
        y_test,
        test_probabilities,
    )

    result["confusion_matrix"] = (
        matrix.tolist()
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset not found. Run "
            "`python ml/generator.py` first."
        )

    df = pd.read_csv(DATA)

    print("\n=== UPAY SENTINEL DATASET ===")
    print(
        f"Total samples: {len(df):,}"
    )

    print("\nClass distribution:")
    print(
        df[TARGET].value_counts()
    )

    print("\nClass share:")
    print(
        df[TARGET]
        .value_counts(normalize=True)
        .round(4)
    )

    # --------------------------------------------------------
    # 2. Build engineered features
    # --------------------------------------------------------

    df = build_features(df)

    required_features = (
        ENHANCED_FEATURES
    )

    missing = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing)
        )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    # --------------------------------------------------------
    # 3. Same 60 / 20 / 20 split
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

    print("\n=== DATA SPLIT ===")
    print(
        f"Training   : {len(X_train):,}"
    )
    print(
        f"Validation : {len(X_val):,}"
    )
    print(
        f"Test       : {len(X_test):,}"
    )

    # --------------------------------------------------------
    # 4. Class imbalance
    # --------------------------------------------------------

    positive_count = int(
        y_train.sum()
    )

    negative_count = int(
        len(y_train) - positive_count
    )

    scale_pos_weight = (
        negative_count
        / max(positive_count, 1)
    )

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # 5. Candidate configurations
    # --------------------------------------------------------

    candidates = [
        {
            "n_estimators": 900,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 900,
            "max_depth": 3,
            "learning_rate": 0.025,
            "min_child_weight": 1,
            "subsample": 0.90,
            "colsample_bytree": 0.95,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 1000,
            "max_depth": 4,
            "learning_rate": 0.025,
            "min_child_weight": 1,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 1000,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 2,
            "subsample": 0.95,
            "colsample_bytree": 0.90,
            "gamma": 0.05,
            "reg_alpha": 0.0,
            "reg_lambda": 1.5,
        },
        {
            "n_estimators": 1000,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 2,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "gamma": 0.05,
            "reg_alpha": 0.05,
            "reg_lambda": 1.5,
        },
        {
            "n_estimators": 1200,
            "max_depth": 3,
            "learning_rate": 0.025,
            "min_child_weight": 2,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "gamma": 0.05,
            "reg_alpha": 0.05,
            "reg_lambda": 1.5,
        },
    ]

    # --------------------------------------------------------
    # 6. Previous-style 13-feature model
    # --------------------------------------------------------

    previous_params = {
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }

    print("\n" + "=" * 70)
    print(
        "REFERENCE MODEL — 13 FEATURES"
    )
    print("=" * 70)

    reference = train_model(
        features=BASE_FEATURES,
        params=previous_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scale_pos_weight=scale_pos_weight,
    )

    reference = evaluate_test(
        reference,
        X_test,
        y_test,
    )

    print(
        f"Validation F1 : "
        f"{reference['validation_f1']:.4f}"
    )

    print(
        f"Test Precision : "
        f"{reference['test_precision']:.4f}"
    )

    print(
        f"Test Recall    : "
        f"{reference['test_recall']:.4f}"
    )

    print(
        f"Test F1        : "
        f"{reference['test_f1']:.4f}"
    )

    print(
        f"ROC-AUC        : "
        f"{reference['test_roc_auc']:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{reference['test_pr_auc']:.4f}"
    )

    # --------------------------------------------------------
    # 7. Enhanced model search
    # --------------------------------------------------------

    enhanced_results = []

    print("\n" + "=" * 70)
    print(
        "ENHANCED MODEL SEARCH — 18 FEATURES"
    )
    print("=" * 70)

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\n[{index}/{len(candidates)}] "
            f"depth={params['max_depth']} "
            f"lr={params['learning_rate']} "
            f"min_child="
            f"{params['min_child_weight']}"
        )

        result = train_model(
            features=ENHANCED_FEATURES,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scale_pos_weight=scale_pos_weight,
        )

        print(
            f"Validation F1 : "
            f"{result['validation_f1']:.4f} | "
            f"Precision : "
            f"{result['validation_precision']:.4f} | "
            f"Recall : "
            f"{result['validation_recall']:.4f} | "
            f"Threshold : "
            f"{result['threshold']:.4f}"
        )

        enhanced_results.append(
            result
        )

    # --------------------------------------------------------
    # 8. Select best enhanced model by validation F1
    # --------------------------------------------------------

    best = max(
        enhanced_results,
        key=lambda result:
            result["validation_f1"],
    )

    best = evaluate_test(
        best,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # 9. Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    comparison = pd.DataFrame(
        [
            {
                "Model": "13-feature reference",
                "Val F1": reference[
                    "validation_f1"
                ],
                "Precision": reference[
                    "test_precision"
                ],
                "Recall": reference[
                    "test_recall"
                ],
                "Test F1": reference[
                    "test_f1"
                ],
                "ROC-AUC": reference[
                    "test_roc_auc"
                ],
                "PR-AUC": reference[
                    "test_pr_auc"
                ],
            },
            {
                "Model": "18-feature enhanced",
                "Val F1": best[
                    "validation_f1"
                ],
                "Precision": best[
                    "test_precision"
                ],
                "Recall": best[
                    "test_recall"
                ],
                "Test F1": best[
                    "test_f1"
                ],
                "ROC-AUC": best[
                    "test_roc_auc"
                ],
                "PR-AUC": best[
                    "test_pr_auc"
                ],
            },
        ]
    )

    print(
        comparison.to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # 10. Best enhanced model details
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "BEST ENHANCED MODEL"
    )
    print("=" * 70)

    print(
        f"Validation F1 : "
        f"{best['validation_f1']:.4f}"
    )

    print(
        f"Test Precision : "
        f"{best['test_precision']:.4f}"
    )

    print(
        f"Test Recall    : "
        f"{best['test_recall']:.4f}"
    )

    print(
        f"Test F1        : "
        f"{best['test_f1']:.4f}"
    )

    print(
        f"ROC-AUC        : "
        f"{best['test_roc_auc']:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{best['test_pr_auc']:.4f}"
    )

    print(
        f"Threshold      : "
        f"{best['threshold']:.4f}"
    )

    print(
        f"Best iteration : "
        f"{best['best_iteration']}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        np.array(
            best["confusion_matrix"]
        )
    )

    print(
        "\nBest parameters:"
    )

    for key, value in best[
        "params"
    ].items():
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # 11. Save selected model
    # --------------------------------------------------------

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        ART / "risk_model.joblib"
    )

    joblib.dump(
        {
            "model": best["model"],
            "features": ENHANCED_FEATURES,
            "threshold": best["threshold"],
            "scale_pos_weight": (
                scale_pos_weight
            ),
            "validation_f1": (
                best["validation_f1"]
            ),
            "test_precision": (
                best["test_precision"]
            ),
            "test_recall": (
                best["test_recall"]
            ),
            "test_f1": (
                best["test_f1"]
            ),
            "test_roc_auc": (
                best["test_roc_auc"]
            ),
            "test_pr_auc": (
                best["test_pr_auc"]
            ),
            "confusion_matrix": (
                best["confusion_matrix"]
            ),
            "feature_engineering": (
                ENGINEERED_FEATURES
            ),
        },
        model_path,
    )

    print(
        f"\nSaved selected model → "
        f"{model_path}"
    )


if __name__ == "__main__":
    main()