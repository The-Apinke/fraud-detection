from pydantic import BaseModel, Field
from typing import Optional


class TransactionInput(BaseModel):
    transaction_amt: float = Field(..., gt=0, description="Transaction amount in USD")
    product_cd: str = Field(..., description="Product code (W, H, C, S, R)")
    card_type: str = Field(..., description="Card type (credit, debit)")
    card_category: str = Field(..., description="Card category (visa, mastercard, etc)")
    p_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    r_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    device_type: Optional[str] = Field(None, description="Device type (desktop, mobile)")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    # Velocity features — computed upstream or passed directly
    txn_count_1h: int = Field(0, ge=0, description="Transactions from this card in last 1 hour")
    txn_count_24h: int = Field(0, ge=0, description="Transactions from this card in last 24 hours")
    amt_vs_mean_ratio: float = Field(1.0, description="Amount / user's 30-day mean amount")
    is_foreign_transaction: bool = Field(False)
    time_since_last_txn_hours: Optional[float] = Field(None, description="Hours since last transaction")


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model's fraud probability (0-1)")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score (0=safe, 100=certain fraud)")
    verdict: str = Field(..., description="APPROVE | REVIEW | BLOCK")
    threshold_used: float
    shap_explanations: list[dict]
    business_impact: dict


class ThresholdAnalysisRequest(BaseModel):
    daily_transaction_volume: int = Field(..., gt=0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    cost_per_false_positive: float = Field(10.0, description="Cost in USD per blocked legitimate transaction")
    cost_per_false_negative: float = Field(150.0, description="Cost in USD per missed fraud")


class ThresholdAnalysisResponse(BaseModel):
    threshold: float
    estimated_false_positives_per_day: int
    estimated_false_negatives_per_day: int
    estimated_daily_fp_cost: float
    estimated_daily_fn_cost: float
    estimated_total_daily_cost: float
    precision: float
    recall: float
    f1: float
