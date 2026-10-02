from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

    The trained models require some derived features that
    are calculated from the raw request values.
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
    # Distance from customer's usual transaction hour
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
        min(
            1.0,
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
        )
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
        # Prepare model features.
        data = prepare_transaction_data(
            transaction
        )

        # Run Sentinel risk pipeline.
        result = risk_service.analyze(
            data
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
        # Prepare model features.
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

        return case

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc