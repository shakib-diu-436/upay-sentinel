from __future__ import annotations
from ml.network_analysis import analyze_network
from ml.account_takeover import analyze_account_takeover

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .database import (
    database_health,
    get_case,
    get_recent_cases,
    init_db,
    save_analysis,
)
from .schemas import (
    InvestigationResponse,
    RiskResponse,
    TransactionRequest,
)
from .services.investigation_service import (
    InvestigationService,
)
from .services.risk_service import RiskService


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="Upay Sentinel API",
    description=(
        "AI-powered financial safety and risk intelligence "
        "API for transaction analysis and investigation support."
    ),
    version="0.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

# Create SQLite database and tables when the API module loads.
init_db()


# ============================================================
# SERVICES
# ============================================================

risk_service = RiskService()
investigation_service = InvestigationService()


# ============================================================
# HELPER — FEATURE PREPARATION
# ============================================================

def prepare_transaction_data(
    transaction: TransactionRequest,
) -> dict:
    """
    Convert the validated API request into the feature
    dictionary required by the trained Sentinel models.

    Raw API fields are used to derive the behavioural features
    required by the trained XGBoost and Isolation Forest models.
    """

    data = transaction.model_dump()

    # --------------------------------------------------------
    # Amount ratio
    # --------------------------------------------------------

    data["amount_ratio"] = (
        data["amount"]
        / max(
            data["avg_amount_30d"],
            1.0,
        )
    )

    # --------------------------------------------------------
    # Distance from usual transaction hour
    # --------------------------------------------------------

    hour_difference = abs(
        data["hour"]
        - data["usual_transaction_hour"]
    )

    data["hour_distance_from_usual"] = min(
        hour_difference,
        24 - hour_difference,
    )

    # --------------------------------------------------------
    # Behavioral deviation score
    # --------------------------------------------------------

    data["behavioral_deviation_score"] = float(
        np.clip(
            (
                0.40
                * min(
                    data["amount_ratio"] / 8.0,
                    2.0,
                )
                + 0.20
                * min(
                    data["hour_distance_from_usual"]
                    / 8.0,
                    2.0,
                )
                + 0.15
                * data["device_changed"]
                + 0.15
                * data["location_changed"]
                + 0.10
                * data["recipient_new"]
            ),
            0.0,
            1.0,
        )
    )

    # ========================================================
    # ENHANCED XGBOOST FEATURES
    # ========================================================

    # Unusual hour: before 06:00
    data["unusual_hour"] = int(
        data["hour"] < 6
    )

    # Log-transformed amount ratio
    data["log_amount_ratio"] = float(
        np.log1p(
            max(
                data["amount_ratio"],
                0.0,
            )
        )
    )

    # Amount is at least 3x customer's average
    data["high_amount_flag"] = int(
        data["amount_ratio"] >= 3.0
    )

    # High transaction velocity
    data["high_velocity_flag"] = int(
        data["transactions_last_1h"] >= 3
    )

    # Count of risk signals
    data["risk_signal_count"] = int(
        data["recipient_new"]
        + data["device_changed"]
        + data["location_changed"]
        + data["high_amount_flag"]
        + data["high_velocity_flag"]
        + data["unusual_hour"]
    )

    return data


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root() -> dict:
    return {
        "project": "Upay Sentinel",
        "status": "running",
        "service": "AI Risk Intelligence API",
        "version": "0.2.0",
        "database": "sqlite",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "service": "upay-sentinel",
    }


# ============================================================
# DATABASE HEALTH
# ============================================================

@app.get("/api/v1/database/health")
def database_status() -> dict:
    """
    Return basic SQLite database statistics.
    """

    try:
        return database_health()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Database health check failed.",
        ) from exc


# ============================================================
# TRANSACTION RISK ANALYSIS
# ============================================================

@app.post(
    "/api/v1/risk/analyze",
    response_model=RiskResponse,
)
def analyze_transaction(
    transaction: TransactionRequest,
) -> RiskResponse:

    try:
        # ----------------------------------------------------
        # Prepare model features
        # ----------------------------------------------------

        data = prepare_transaction_data(
            transaction
        )

        # ----------------------------------------------------
        # Run Sentinel risk pipeline
        # ----------------------------------------------------

        result = risk_service.analyze(
            data
        )

        # ----------------------------------------------------
        # Persist transaction + risk assessment
        # ----------------------------------------------------

        save_analysis(
            transaction=data,
            analysis=result,
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# INVESTIGATION ANALYSIS
# ============================================================

@app.post(
    "/api/v1/investigation/analyze",
    response_model=InvestigationResponse,
)
def investigate_transaction(
    transaction: TransactionRequest,
) -> InvestigationResponse:

    try:
        # ----------------------------------------------------
        # Prepare model features
        # ----------------------------------------------------

        data = prepare_transaction_data(
            transaction
        )

        # ----------------------------------------------------
        # Run complete Sentinel risk pipeline
        # ----------------------------------------------------

        analysis = risk_service.analyze(
            data
        )

        # ----------------------------------------------------
        # Build evidence-grounded investigation case
        # ----------------------------------------------------

        case = investigation_service.build_case(
            transaction=data,
            analysis=analysis,
        )

        # ----------------------------------------------------
        # Persist complete investigation
        # ----------------------------------------------------

        save_analysis(
            transaction=data,
            analysis=analysis,
            case=case,
        )

        return case

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# RECENT INVESTIGATION CASES
# ============================================================

@app.get("/api/v1/cases")
def recent_cases(
    limit: int = 20,
) -> dict:
    """
    Return the most recently created investigation cases.
    """

    try:
        return {
            "count": len(
                get_recent_cases(limit)
            ),
            "cases": get_recent_cases(limit),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load investigation cases.",
        ) from exc


# ============================================================
# SINGLE INVESTIGATION CASE
# ============================================================

@app.get("/api/v1/cases/{case_id}")
def single_case(
    case_id: int,
) -> dict:
    """
    Return one persisted investigation case.
    """

    try:
        case = get_case(case_id)

        if case is None:
            raise HTTPException(
                status_code=404,
                detail="Investigation case not found.",
            )

        return case

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load investigation case.",
        ) from exc