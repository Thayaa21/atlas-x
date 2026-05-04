"""
ATLAS-X API – Pydantic request/response schemas.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Predict ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    transaction_id: str
    card_id:        str
    amount:         float = Field(gt=0)
    features:       Dict[str, Any]
    is_fraud_label: Optional[bool] = None   # ground truth from Kafka producer
    store_embedding: bool = False            # only True when full features are present (Kafka stream)


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
    # Optional overrides — when provided, skip re-scoring and use these values
    # directly in the LLM prompt. Use this when the frontend already has the
    # correct fraud_probability and decision from a prior /predict call.
    fraud_probability_override: Optional[float] = None
    decision_override:          Optional[str]   = None
    graph_risk_override:        Optional[float] = None
    customer_segment_override:  Optional[str]   = None


class ExplainResponse(BaseModel):
    transaction_id:     str
    shap_values:        Dict[str, float]    # top-10 feature → SHAP value
    graph_explanation:  str
    decision_reasoning: str


# ── Dual LLM Explain (Feature 7) ─────────────────────────────────────────────

class DualExplainResponse(BaseModel):
    transaction_id:      str
    shap_values:         Dict[str, float]
    graph_explanation:   str
    decision_reasoning:  str
    qwen_explanation:    Optional[str] = None
    openai_explanation:  Optional[str] = None
    model_used:          str           # "qwen" | "openai" | "both"
    generation_time_ms:  float


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


# ── Event Sourcing (Feature 1) ────────────────────────────────────────────────

class EventRecord(BaseModel):
    event_type:  str
    data:        Dict[str, Any]
    timestamp:   str


class EventHistoryResponse(BaseModel):
    transaction_id: str
    events:         List[EventRecord]
    total:          int


# ── Human-in-the-Loop (Feature 3) ────────────────────────────────────────────

class FlaggedTransaction(BaseModel):
    transaction_id:    str
    fraud_probability: float
    graph_risk_score:  float
    customer_segment:  str
    decision:          str
    latency_ms:        float
    created_at:        str
    minutes_waiting:   float
    flag_reason:       str


class FlaggedResponse(BaseModel):
    flagged:    List[FlaggedTransaction]
    total:      int


# ── MCP Integration (Feature 4) ──────────────────────────────────────────────

class MCPSlackRequest(BaseModel):
    transaction_id: str
    message:        str
    channel:        str = "#fraud-alerts"


class MCPSlackResponse(BaseModel):
    status:         str
    mock:           bool
    channel:        str
    transaction_id: str
    message:        str
    timestamp:      str


# ── pgvector Similarity (Feature 6) ──────────────────────────────────────────

class SimilarTransaction(BaseModel):
    transaction_id:    str
    similarity_score:  float
    is_fraud:          bool
    created_at:        str


class SimilarityResponse(BaseModel):
    query_transaction_id: str
    similar_transactions: List[SimilarTransaction]
    total:                int
