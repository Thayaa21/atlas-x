"""
ATLAS-X Fraud Detection API
────────────────────────────
v1 endpoints  (production):
  POST /api/v1/predict              – score a transaction
  POST /api/v1/explain              – SHAP top-10 explanation
  GET  /api/v1/graph/rings          – list fraud rings from Neo4j
  GET  /api/v1/graph/card/{card_id} – card graph connections
  GET  /api/v1/health               – readiness probe
  GET  /api/v1/stats                – runtime statistics

Legacy endpoints (backward-compat):
  POST /api/predict
  GET  /api/graph/rings
  GET  /api/metrics
  GET  /healthz
  GET  /api/metrics/prometheus      (Prometheus scrape target)
  WS   /ws/alerts                   (Kafka → WebSocket bridge)
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Load .env from repo root so OPENAI_API_KEY etc. are always available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator

from src.api.model_loader import ModelLoader
from src.api.monitoring import StatsTracker
from src.api.schemas import (
    CardGraphResponse, ExplainRequest, ExplainResponse,
    HealthResponse, PredictRequest, PredictResponse,
    RingsResponse, FraudRing, StatsResponse,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── Singletons ────────────────────────────────────────────────────────────────
loader  = ModelLoader()
tracker = StatsTracker()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ATLAS-X Fraud Detection API",
    version="2.0.0",
    description="XGBoost v4 + Neo4j graph layer",
)
Instrumentator().instrument(app).expose(app, endpoint="/api/metrics/prometheus")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup() -> None:
    loader.load_all()


# ── Decision logic ────────────────────────────────────────────────────────────

def _make_decision(fraud_prob: float, threshold: float,
                   graph_risk: float, in_ring: bool) -> str:
    """
    Three-tier decision:
      BLOCK  – model says fraud (prob ≥ segment threshold)
      FLAG   – borderline or confirmed ring member with elevated prob
      APPROVE– low risk
    """
    if fraud_prob >= threshold:
        return "BLOCK"
    # FLAG if prob is elevated AND card is in a fraud ring
    flag_floor = threshold * 0.4
    if fraud_prob >= flag_floor and (in_ring or graph_risk >= 0.5):
        return "FLAG"
    return "APPROVE"


def _confidence(fraud_prob: float, decision: str) -> float:
    if decision == "BLOCK":
        return round(fraud_prob, 4)
    if decision == "APPROVE":
        return round(1.0 - fraud_prob, 4)
    # FLAG: distance from center
    return round(abs(fraud_prob - 0.5) + 0.5, 4)


# ── Core predict helper ───────────────────────────────────────────────────────

def _predict_core(transaction_id: str, card_id: str,
                  features: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Feature prep
    X        = loader.prepare_features(features)
    segment  = loader.segment_for(features)
    threshold = loader.thresholds.get(segment, 0.5)

    # XGBoost inference
    fraud_prob   = float(loader.model.predict_proba(X)[:, 1][0])

    # Graph risk – prefer precomputed column, fall back to live Neo4j check
    graph_risk   = float(features.get("graph_risk_score") or 0.0)
    in_ring      = False
    ring_details: Optional[dict] = None

    if loader.ring_checker is not None:
        try:
            ring_details = loader.ring_checker.check(card_id)
            in_ring      = bool(ring_details.get("in_fraud_ring", False))
            if in_ring:
                graph_risk = max(graph_risk,
                                 float(ring_details.get("graph_risk_score", 0.0)))
        except Exception:
            pass  # Neo4j hiccup — continue without ring override

    decision   = _make_decision(fraud_prob, threshold, graph_risk, in_ring)
    confidence = _confidence(fraud_prob, decision)
    latency_ms = (time.perf_counter() - t0) * 1000

    tracker.record(
        decision=decision,
        latency_ms=latency_ms,
        fraud_probability=fraud_prob,
        extra={
            "transaction_id":   transaction_id,
            "graph_risk_score": round(graph_risk, 4),
            "customer_segment": segment,
            "confidence":       confidence,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "transaction_id":    transaction_id,
        "fraud_probability": round(fraud_prob, 4),
        "graph_risk_score":  round(graph_risk, 4),
        "customer_segment":  segment,
        "decision":          decision,
        "confidence":        confidence,
        "latency_ms":        round(latency_ms, 2),
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "_ring_details":     ring_details,     # internal – stripped from v1 response
    }


# ══════════════════════════════════════════════════════════════════════════════
#  API v1 endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Score a transaction.  The `features` dict must contain the full
    pre-computed feature set (same columns used to train XGBoost v4).
    """
    result = _predict_core(req.transaction_id, req.card_id, req.features)
    return PredictResponse(
        transaction_id    =result["transaction_id"],
        fraud_probability =result["fraud_probability"],
        graph_risk_score  =result["graph_risk_score"],
        customer_segment  =result["customer_segment"],
        decision          =result["decision"],
        confidence        =result["confidence"],
        latency_ms        =result["latency_ms"],
        timestamp         =result["timestamp"],
    )


@app.post("/api/v1/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """
    SHAP top-10 feature attribution for a transaction.
    Pass `features` (same dict as /predict) alongside the transaction_id.
    """
    if req.features is None:
        raise HTTPException(
            status_code=422,
            detail="features dict is required for explanation.",
        )

    X         = loader.prepare_features(req.features)
    explainer = loader.get_shap_explainer()

    import shap
    shap_vals = explainer(X)
    vals      = shap_vals[0].values
    names     = list(X.columns)

    # Top-10 by |SHAP|
    top_idx   = np.argsort(np.abs(vals))[::-1][:10]
    top_shap  = {names[i]: round(float(vals[i]), 5) for i in top_idx}

    # Graph explanation text
    ring_info = ""
    if loader.ring_checker is not None:
        try:
            ring = loader.ring_checker.check(
                req.features.get("card_id", "unknown")
            )
            if ring.get("in_fraud_ring"):
                ring_info = (
                    f"Card shares device with "
                    f"{ring.get('device_frauds', 0)} confirmed fraud cards."
                )
        except Exception:
            pass
    graph_explanation = ring_info or "No fraud ring connection detected."

    # Decision reasoning
    fp        = float(loader.model.predict_proba(X)[:, 1][0])
    segment   = loader.segment_for(req.features)
    threshold = loader.thresholds.get(segment, 0.5)
    decision  = _make_decision(fp, threshold, 0.0, bool(ring_info))
    reasoning = (
        f"fraud_prob={fp:.3f} vs threshold={threshold:.3f} "
        f"(segment={segment}) → {decision}"
    )
    if ring_info:
        reasoning += f";  {ring_info}"

    return ExplainResponse(
        transaction_id     =req.transaction_id,
        shap_values        =top_shap,
        graph_explanation  =graph_explanation,
        decision_reasoning =reasoning,
    )


# ── /api/v1/graph/rings ───────────────────────────────────────────────────────

_RING_QUERY = """
MATCH (d:{node_label})
WHERE d.fraud_txn > 0
WITH d
MATCH (c:Card)-[:{rel_type}]->(d)
WITH d, count(DISTINCT c) AS card_count,
     sum(CASE WHEN c.fraud_txn > 0 THEN 1 ELSE 0 END) AS fraud_count
WHERE card_count >= $min_cards AND fraud_count >= $min_frauds
RETURN d.{id_prop} AS ring_id, card_count, fraud_count,
       toFloat(fraud_count) / card_count AS fraud_rate
ORDER BY fraud_count DESC
LIMIT 200
"""

_NEO4J_CSV_DIR = Path(__file__).resolve().parents[2] / "data" / "neo4j"


def _rings_from_csv(min_cards: int, min_frauds: int) -> list[dict]:
    """Compute fraud rings directly from the exported CSV files (no Neo4j needed)."""
    import pandas as pd
    rel_path = _NEO4J_CSV_DIR / "relationships.csv"
    if not rel_path.exists():
        return []
    rel = pd.read_csv(rel_path)
    # normalise boolean is_fraud column
    rel["is_fraud"] = rel["is_fraud"].map(
        lambda v: v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
    )
    rings: list[dict] = []
    specs = [
        ("device_id", "device"),
        ("email_id",  "email"),
        ("addr_id",   "address"),
    ]
    for col, rtype in specs:
        sub = rel[["card_id", col, "is_fraud"]].dropna(subset=[col])
        grp = sub.groupby(col).agg(
            card_count=("card_id", "nunique"),
            fraud_count=("is_fraud", "sum"),
        ).reset_index()
        grp = grp[
            (grp["card_count"] >= min_cards) & (grp["fraud_count"] >= min_frauds)
        ]
        for _, row in grp.iterrows():
            card_count  = int(row["card_count"])
            fraud_count = int(row["fraud_count"])
            rings.append({
                "ring_type":  rtype,
                "ring_id":    str(row[col]),
                "card_count": card_count,
                "fraud_count": fraud_count,
                "fraud_rate": round(fraud_count / card_count, 4),
            })
    rings.sort(key=lambda x: -x["fraud_count"])
    return rings


def _query_rings(min_cards: int, min_frauds: int) -> list[dict]:
    if loader.ring_checker is None:
        return _rings_from_csv(min_cards, min_frauds)
    rings = []
    node_specs = [
        ("Device",  "USED_DEVICE",  "device_id", "device"),
        ("Email",   "HAS_EMAIL",    "email_id",  "email"),
        ("Address", "BILLING_ADDR", "addr_id",   "address"),
    ]
    from neo4j import GraphDatabase
    import os
    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pw   = os.getenv("NEO4J_PASSWORD", "password123")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        for label, rel, id_prop, rtype in node_specs:
            q = _RING_QUERY.format(
                node_label=label, rel_type=rel, id_prop=id_prop
            )
            with driver.session() as s:
                for r in s.run(q, min_cards=min_cards, min_frauds=min_frauds):
                    rings.append({
                        "ring_type":  rtype,
                        "ring_id":    str(r["ring_id"]),
                        "card_count": int(r["card_count"]),
                        "fraud_count":int(r["fraud_count"]),
                        "fraud_rate": round(float(r["fraud_rate"]), 4),
                    })
        driver.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}")
    rings.sort(key=lambda x: -x["fraud_count"])
    return rings


@app.get("/api/v1/graph/rings", response_model=RingsResponse)
def graph_rings(
    min_cards:  int = Query(default=3, ge=2),
    min_frauds: int = Query(default=2, ge=1),
):
    """List fraud rings from Neo4j, ordered by fraud card count."""
    rings = _query_rings(min_cards, min_frauds)
    return RingsResponse(
        rings=[FraudRing(**r) for r in rings],
        total=len(rings),
    )


# ── /api/v1/graph/card/{card_id} ──────────────────────────────────────────────

@app.get("/api/v1/graph/card/{card_id}", response_model=CardGraphResponse)
def graph_card(card_id: str):
    """Return the graph risk profile for a single card."""
    if loader.ring_checker is None:
        raise HTTPException(status_code=503, detail="Neo4j unavailable.")
    try:
        result = loader.ring_checker.check(card_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return CardGraphResponse(
        card_id         =card_id,
        connected_frauds=result.get("connected_frauds", 0),
        device_frauds   =result.get("device_frauds",   0),
        email_frauds    =result.get("email_frauds",    0),
        address_frauds  =result.get("address_frauds",  0),
        graph_risk_score=result.get("graph_risk_score",0.0),
        in_fraud_ring   =result.get("in_fraud_ring",   False),
        ring_type       =result.get("ring_type",       "none"),
    )


# ── /api/v1/health ────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status          ="healthy",
        xgboost_loaded  =getattr(loader, "_loaded", False),
        neo4j_connected =loader.neo4j_ok,
        uptime_seconds  =round(tracker.uptime(), 1),
    )


# ── /api/v1/stats ─────────────────────────────────────────────────────────────

@app.get("/api/v1/recent")
def recent_transactions():
    """Last 50 scored transactions (lightweight metadata, no features)."""
    return {"transactions": tracker.recent()}


@app.get("/api/v1/stats", response_model=StatsResponse)
def stats():
    s = tracker.snapshot()
    return StatsResponse(**s)


# ── /api/v1/explain/llm ───────────────────────────────────────────────────────

class LLMExplainRequest(BaseModel):
    transaction_id:    str
    fraud_probability: float
    decision:          str
    customer_segment:  str
    graph_risk_score:  float
    shap_values:       Dict[str, float]
    graph_explanation: str


class LLMExplainResponse(BaseModel):
    transaction_id: str
    explanation:    str


@app.post("/api/v1/explain/llm", response_model=LLMExplainResponse)
def explain_llm(req: LLMExplainRequest):
    """
    Natural-language explanation of a fraud decision, generated by GPT-4o.
    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured.")

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=503, detail="openai package not installed. Run: pip install openai")

    top_features = sorted(req.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    feature_lines = "\n".join(
        f"  - {name}: SHAP={val:+.4f}" for name, val in top_features
    )

    prompt = f"""You are an expert fraud analyst. Explain the following fraud detection result clearly and concisely.

Transaction ID: {req.transaction_id}
Decision: {req.decision}
Fraud Probability: {req.fraud_probability:.4f}
Customer Segment: {req.customer_segment}
Graph Risk Score: {req.graph_risk_score:.4f}
Graph Context: {req.graph_explanation}

Top contributing features (SHAP values):
{feature_lines}

Write a 3-4 sentence explanation for a fraud analyst. Lead with the decision and probability. Mention the top 2-3 contributing factors. End with a recommendation."""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    explanation = response.choices[0].message.content.strip()

    return LLMExplainResponse(
        transaction_id=req.transaction_id,
        explanation=explanation,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy endpoints (backward-compat)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/predict")
def api_predict_legacy(payload: Dict[str, Any]):
    """
    Legacy endpoint — wraps the v1 predict logic.
    Expects the full raw transaction dict (same format as before).
    """
    tx = payload.get("transaction", payload)
    # Build a minimal features dict from whatever fields are present
    features = dict(tx)
    txn_id   = str(tx.get("TransactionID", "legacy"))
    card_id  = (
        f"{tx.get('card1','?')}-"
        f"{tx.get('card2','NA')}-"
        f"{tx.get('card3','NA')}"
    )
    try:
        result = _predict_core(txn_id, card_id, features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    result.pop("_ring_details", None)
    return JSONResponse(result)


@app.get("/api/graph/rings")
def api_graph_rings_legacy(
    min_cards:  int = Query(default=3),
    min_frauds: int = Query(default=2),
):
    rings = _query_rings(min_cards, min_frauds)
    return {"rings": rings, "total": len(rings)}


@app.get("/api/metrics")
def api_metrics():
    import psutil, os as _os
    proc   = psutil.Process(_os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    return {
        "uptime_seconds": round(tracker.uptime(), 1),
        "rss_memory_mb":  round(mem_mb, 1),
        "predictions":    tracker.snapshot(),
    }


# ── WebSocket: Kafka → browser bridge ────────────────────────────────────────

@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """Real-time fraud alert feed (Kafka consumer → WebSocket)."""
    await websocket.accept()
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    topic           = os.getenv("KAFKA_TOPIC_FRAUD_ALERTS", "fraud-alerts")
    group_id        = os.getenv("KAFKA_WS_GROUP_ID",        "fraud-ws-group")
    try:
        from aiokafka import AIOKafkaConsumer
        consumer = AIOKafkaConsumer(
            topic, bootstrap_servers=kafka_bootstrap,
            group_id=group_id, auto_offset_reset="latest",
        )
        await consumer.start()
        try:
            while True:
                msg = await consumer.getone()
                await websocket.send_text(msg.value.decode("utf-8"))
        finally:
            await consumer.stop()
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
