"""
ATLAS-X API – Pydantic request/response schemas.
"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ── Predict ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    transaction_id: str
    card_id:        str
    amount:         float = Field(gt=0)
    features:       Dict[str, Any]  # full pre-computed feature dict (498 cols)


class PredictResponse(BaseModel):
    transaction_id:    str
    fraud_probability: float
    graph_risk_score:  float
    customer_segment:  str
    decision:          str          # APPROVE | FLAG | BLOCK
    confidence:        float
    latency_ms:        float
    timestamp:         str


# ── Explain ───────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    transaction_id: str
    features:       Optional[Dict[str, Any]] = None  # required if no cached pred


class ExplainResponse(BaseModel):
    transaction_id:     str
    shap_values:        Dict[str, float]    # top-10 feature → SHAP value
    graph_explanation:  str
    decision_reasoning: str


# ── Graph ─────────────────────────────────────────────────────────────────────

class FraudRing(BaseModel):
    ring_type:   str        # device | email | address
    ring_id:     str
    card_count:  int
    fraud_count: int
    fraud_rate:  float


class RingsResponse(BaseModel):
    rings: list[FraudRing]
    total: int


class CardGraphResponse(BaseModel):
    card_id:          str
    connected_frauds: int
    device_frauds:    int
    email_frauds:     int
    address_frauds:   int
    graph_risk_score: float
    in_fraud_ring:    bool
    ring_type:        str


# ── Health / Stats ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:          str
    xgboost_loaded:  bool
    neo4j_connected: bool
    uptime_seconds:  float


class StatsResponse(BaseModel):
    total_predictions: int
    fraud_rate:        float
    avg_latency_ms:    float
    decisions:         Dict[str, int]
