from __future__ import annotations

from typing import Any

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.database import (
    database_health,
    get_case,
    get_recent_cases,
    init_db,
    save_analysis,
)

from backend.app.schemas import (
    InvestigationResponse,
    RiskResponse,
    TransactionRequest,
)

from backend.app.services.investigation_service import (
    InvestigationService,
)

from backend.app.services.risk_service import (
    RiskService,
)

from ml.account_takeover import (
    analyze_account_takeover as run_account_takeover_analysis,
)

from ml.network_analysis import (
    analyze_network,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Upay Sentinel API",
    version="0.3.0",
    description=(
        "AI-powered financial safety and investigation "
        "intelligence layer for suspicious wallet activity."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
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
# DATABASE INITIALIZATION
# ============================================================

init_db()


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_transaction_data(
    request: TransactionRequest,
) -> dict[str, Any]:
    """
    Convert API request into the feature dictionary expected
    by the trained XGBoost and Isolation Forest models.

    The derived features must remain consistent between
    training and serving.
    """

    amount = float(
        request.amount
    )

    avg_amount_30d = float(
        request.avg_amount_30d
    )

    usual_transaction_hour = int(
        request.usual_transaction_hour
    )

    hour = int(
        request.hour
    )

    transactions_last_1h = int(
        request.transactions_last_1h
    )

    transactions_last_24h = int(
        request.transactions_last_24h
    )

    recipient_new = int(
        request.recipient_new
    )

    device_changed = int(
        request.device_changed
    )

    location_changed = int(
        request.location_changed
    )

    # --------------------------------------------------------
    # Amount ratio
    # --------------------------------------------------------

    amount_ratio = (
        amount
        / max(avg_amount_30d, 1.0)
    )

    # --------------------------------------------------------
    # Circular hour distance
    #
    # Example:
    # 23:00 vs 01:00 => 2 hours
    # not 22 hours
    # --------------------------------------------------------

    hour_distance_from_usual = abs(
        hour
        - usual_transaction_hour
    )

    hour_distance_from_usual = min(
        hour_distance_from_usual,
        24 - hour_distance_from_usual,
    )

    # --------------------------------------------------------
    # Behavioral deviation score
    # Same formula used for generated behavioral context.
    # --------------------------------------------------------

    behavioral_deviation_score = float(
        np.clip(
            0.40
            * min(
                amount_ratio / 8.0,
                2.0,
            )
            + 0.20
            * min(
                hour_distance_from_usual / 8.0,
                2.0,
            )
            + 0.15
            * device_changed
            + 0.15
            * location_changed
            + 0.10
            * recipient_new,
            0.0,
            1.0,
        )
    )

    # ========================================================
    # ENHANCED FEATURES FOR CURRENT 18-FEATURE MODEL
    # ========================================================

    unusual_hour = int(
        hour < 6
    )

    log_amount_ratio = float(
        np.log1p(
            max(
                amount_ratio,
                0.0,
            )
        )
    )

    high_amount_flag = int(
        amount_ratio >= 3.0
    )

    high_velocity_flag = int(
        transactions_last_1h >= 3
    )

    risk_signal_count = int(
        recipient_new
        + device_changed
        + location_changed
        + high_amount_flag
        + high_velocity_flag
        + unusual_hour
    )

    # ========================================================
    # RETURN COMPLETE TRANSACTION CONTEXT
    # ========================================================

    return {
        "transaction_id": (
            request.transaction_id
        ),

        "customer_id": (
            request.customer_id
        ),

        "recipient_id": (
            request.recipient_id
        ),

        "timestamp": (
            request.timestamp
        ),

        "amount": amount,

        "hour": hour,

        "recipient_new": (
            recipient_new
        ),

        "device_changed": (
            device_changed
        ),

        "location_changed": (
            location_changed
        ),

        "transactions_last_1h": (
            transactions_last_1h
        ),

        "transactions_last_24h": (
            transactions_last_24h
        ),

        "avg_amount_30d": (
            avg_amount_30d
        ),

        "usual_transaction_hour": (
            usual_transaction_hour
        ),

        "account_age_days": int(
            request.account_age_days
        ),

        # Derived features
        "amount_ratio": float(
            amount_ratio
        ),

        "hour_distance_from_usual": float(
            hour_distance_from_usual
        ),

        "behavioral_deviation_score": float(
            behavioral_deviation_score
        ),

        # Enhanced XGBoost features
        "unusual_hour": unusual_hour,

        "log_amount_ratio": (
            log_amount_ratio
        ),

        "high_amount_flag": (
            high_amount_flag
        ),

        "high_velocity_flag": (
            high_velocity_flag
        ),

        "risk_signal_count": (
            risk_signal_count
        ),
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Upay Sentinel API",
        "version": "0.3.0",
        "status": "online",
        "purpose": (
            "AI financial safety, risk scoring, "
            "network intelligence and investigation support."
        ),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "upay-sentinel",
        "version": "0.3.0",
    }


# ============================================================
# DATABASE HEALTH
# ============================================================

@app.get("/api/v1/database/health")
def db_health():
    try:
        return database_health()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Database health check failed.",
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

@app.post(
    "/api/v1/risk/analyze",
    response_model=RiskResponse,
)
def analyze_risk(
    request: TransactionRequest,
):
    try:
        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        transaction = (
            prepare_transaction_data(
                request
            )
        )

        # ----------------------------------------------------
        # Model inference + fusion
        # ----------------------------------------------------

        analysis = (
            risk_service.analyze(
                transaction
            )
        )

        # ----------------------------------------------------
        # Persist transaction + risk assessment
        #
        # This endpoint does not create a full
        # investigation case.
        # ----------------------------------------------------

        save_analysis(
            transaction=transaction,
            analysis=analysis,
        )

        return analysis

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Risk analysis failed.",
        )


# ============================================================
# INVESTIGATION ANALYSIS
# ============================================================

@app.post(
    "/api/v1/investigation/analyze",
    response_model=InvestigationResponse,
)
def analyze_investigation(
    request: TransactionRequest,
):
    try:
        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        transaction = (
            prepare_transaction_data(
                request
            )
        )

        # ----------------------------------------------------
        # AI risk analysis
        # ----------------------------------------------------

        analysis = (
            risk_service.analyze(
                transaction
            )
        )

        # ----------------------------------------------------
        # Investigation case
        # ----------------------------------------------------

        case = (
            investigation_service.build_case(
                transaction=transaction,
                analysis=analysis,
            )
        )

        # ----------------------------------------------------
        # Persist everything
        # ----------------------------------------------------

        save_analysis(
            transaction=transaction,
            analysis=analysis,
            case=case,
        )

        return case

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Investigation analysis failed.",
        )


# ============================================================
# RECENT INVESTIGATION CASES
# ============================================================

@app.get(
    "/api/v1/cases",
)
def recent_cases(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
):
    """
    Return recent investigation cases
    from the persistent SQLite database.
    """

    try:
        cases = get_recent_cases(
            limit=limit
        )

        return {
            "cases": cases,
            "count": len(cases),
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not load recent cases.",
        )


# ============================================================
# SINGLE CASE
# ============================================================

@app.get(
    "/api/v1/cases/{case_id}",
)
def single_case(
    case_id: str,
):
    try:
        case = get_case(
            case_id
        )

        if case is None:
            raise HTTPException(
                status_code=404,
                detail="Investigation case not found.",
            )

        return {
            "case": case,
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not load investigation case.",
        )


# ============================================================
# SUSPICIOUS NETWORK / MULE ANALYSIS
# ============================================================

@app.post(
    "/api/v1/network/analyze",
)
def analyze_transaction_network(
    request: dict[str, Any],
):
    """
    Analyze customer-recipient relationships and
    identify suspicious network / possible mule patterns.

    Supported identifiers:
    - transaction_id
    - customer_id
    - recipient_id
    """

    try:
        result = analyze_network(
            customer_id=request.get(
                "customer_id"
            ),

            recipient_id=request.get(
                "recipient_id"
            ),

            transaction_id=request.get(
                "transaction_id"
            ),
        )

        return result

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Network analysis failed.",
        )


# ============================================================
# ACCOUNT TAKEOVER INTELLIGENCE
# ============================================================

@app.post(
    "/api/v1/account-takeover/analyze",
)
def analyze_account_takeover(
    request: dict[str, Any],
):
    """
    Detect account-takeover-like behavioural patterns
    using device, location, timing, recipient, amount
    and velocity signals.

    This is an investigation-support score and does not
    autonomously block or deny financial activity.
    """

    try:
        result = (
            run_account_takeover_analysis(
                transaction=request
            )
        )

        return result

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Account takeover analysis failed."
            ),
        )


# ============================================================
# OPTIONAL COMBINED INTELLIGENCE ENDPOINT
# ============================================================

@app.post(
    "/api/v1/intelligence/analyze",
)
def analyze_extended_intelligence(
    request: TransactionRequest,
):
    """
    Run the normal Sentinel investigation and extend it
    with network and account-takeover intelligence.

    This gives the frontend a single endpoint when needed.
    """

    try:
        # ----------------------------------------------------
        # 1. Prepare transaction
        # ----------------------------------------------------

        transaction = (
            prepare_transaction_data(
                request
            )
        )

        # ----------------------------------------------------
        # 2. Core Sentinel analysis
        # ----------------------------------------------------

        analysis = (
            risk_service.analyze(
                transaction
            )
        )

        # ----------------------------------------------------
        # 3. Investigation case
        # ----------------------------------------------------

        case = (
            investigation_service.build_case(
                transaction=transaction,
                analysis=analysis,
            )
        )

        # ----------------------------------------------------
        # 4. Network intelligence
        # ----------------------------------------------------

        network_result = analyze_network(
            customer_id=request.customer_id,
            recipient_id=request.recipient_id,
            transaction_id=request.transaction_id,
        )

        # ----------------------------------------------------
        # 5. Account takeover intelligence
        # ----------------------------------------------------

        account_takeover_result = (
            run_account_takeover_analysis(
                transaction=transaction
            )
        )

        # ----------------------------------------------------
        # 6. Persist core case
        # ----------------------------------------------------

        save_analysis(
            transaction=transaction,
            analysis=analysis,
            case=case,
        )

        # ----------------------------------------------------
        # 7. Return combined response
        # ----------------------------------------------------

        return {
            **case,

            "network_intelligence": (
                network_result
            ),

            "account_takeover_intelligence": (
                account_takeover_result
            ),
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Extended intelligence analysis failed."
            ),
        )