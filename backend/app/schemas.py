from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================================
# TRANSACTION REQUEST
# ============================================================

class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    recipient_id: str
    timestamp: str

    amount: float = Field(gt=0)

    recipient_new: int = Field(
        ge=0,
        le=1,
    )

    device_changed: int = Field(
        ge=0,
        le=1,
    )

    location_changed: int = Field(
        ge=0,
        le=1,
    )

    transactions_last_1h: int = Field(
        ge=0,
    )

    transactions_last_24h: int = Field(
        ge=0,
    )

    avg_amount_30d: float = Field(
        gt=0,
    )

    usual_transaction_hour: int = Field(
        ge=0,
        le=23,
    )

    hour: int = Field(
        ge=0,
        le=23,
    )

    account_age_days: int = Field(
        ge=0,
    )


# ============================================================
# RISK REASON
# ============================================================

class RiskReason(BaseModel):
    feature: str
    shap_value: float
    direction: str


# ============================================================
# RISK RESPONSE
# ============================================================

class RiskResponse(BaseModel):
    transaction_risk: float
    anomaly_score: float
    final_risk_score: float
    risk_level: str

    recommended_action: str

    model_threshold: float

    reasons: list[RiskReason]


# ============================================================
# INVESTIGATION EVIDENCE
# ============================================================

class InvestigationEvidence(BaseModel):
    type: str
    severity: str
    title: str
    detail: str


# ============================================================
# INVESTIGATION MODEL REASON
# ============================================================

class InvestigationModelReason(BaseModel):
    feature: str
    label: str
    shap_value: float
    direction: str


# ============================================================
# INVESTIGATION RESPONSE
# ============================================================

class InvestigationResponse(BaseModel):
    case_status: str

    transaction_id: str
    customer_id: str
    recipient_id: str
    timestamp: str

    risk_level: str

    final_risk_score: float
    transaction_risk: float
    anomaly_score: float

    summary: str

    evidence: list[InvestigationEvidence]

    model_reasons: list[InvestigationModelReason]

    recommended_human_action: str

    notice: str