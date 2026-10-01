from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    amount: float = Field(gt=0)

    recipient_new: int = Field(ge=0, le=1)
    device_changed: int = Field(ge=0, le=1)
    location_changed: int = Field(ge=0, le=1)

    transactions_last_1h: int = Field(ge=0)
    transactions_last_24h: int = Field(ge=0)

    avg_amount_30d: float = Field(gt=0)
    usual_transaction_hour: int = Field(ge=0, le=23)
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(ge=0)


class RiskReason(BaseModel):
    feature: str
    shap_value: float
    direction: str


class RiskResponse(BaseModel):
    transaction_risk: float
    anomaly_score: float
    final_risk_score: float
    risk_level: str
    recommended_action: str
    model_threshold: float
    reasons: list[RiskReason]