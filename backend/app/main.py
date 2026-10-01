from __future__ import annotations

from fastapi import FastAPI, HTTPException



from .schemas import (
    RiskResponse,
    TransactionRequest,
)
from .services.risk_service import RiskService
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Upay Sentinel API",
    description=(
        "AI-powered transaction risk and "
        "behavioral anomaly analysis API."
    ),
    version="0.1.0",
)
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

risk_service = RiskService()


@app.get("/")
def root() -> dict:
    return {
        "project": "Upay Sentinel",
        "status": "running",
        "service": "AI Risk Engine",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "service": "upay-sentinel",
    }


@app.post(
    "/api/v1/risk/analyze",
    response_model=RiskResponse,
)
def analyze_transaction(
    transaction: TransactionRequest,
) -> RiskResponse:

    try:

        # Convert validated Pydantic object to dictionary.
        data = transaction.model_dump()

        # Derived feature required by our trained model.
        data["amount_ratio"] = (
            data["amount"]
            / max(
                data["avg_amount_30d"],
                1.0,
            )
        )

        hour_difference = abs(
            data["hour"]
            - data["usual_transaction_hour"]
        )

        data["hour_distance_from_usual"] = min(
            hour_difference,
            24 - hour_difference,
        )

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

        result = risk_service.analyze(
            data
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc