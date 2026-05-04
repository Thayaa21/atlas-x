"""
ATLAS-X Fraud Detection API  v3.0.0
─────────────────────────────────────
v1 endpoints  (production):
  POST /api/v1/predict                    – score a transaction
  POST /api/v1/explain                    – SHAP + dual-LLM explanation (Feature 7)
  GET  /api/v1/events/{transaction_id}    – event-sourced audit trail (Feature 1)
  GET  /api/v1/flagged                    – human-in-the-loop review queue (Feature 3)
  POST /api/v1/mcp/slack/alert            – MCP enterprise integration (Feature 4)
  GET  /api/v1/similar/{transaction_id}   – pgvector similarity search (Feature 6)
  GET  /api/v1/graph/rings                – list fraud rings from Neo4j
  GET  /api/v1/graph/card/{card_id}       – card graph connections
  GET  /api/v1/health                     – readiness probe
  GET  /api/v1/stats                      – runtime statistics
  GET  /api/v1/recent                     – last 50 scored transactions

Legacy endpoints (backward-compat):
  POST /api/predict
  GET  /api/graph/rings
  GET  /api/metrics
  GET  /healthz
  GET  /api/metrics/prometheus      (Prometheus scrape target)
  WS   /ws/alerts                   (Kafka → WebSocket bridge)
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from repo root so OPENAI_API_KEY etc. are always available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator

from src.api.model_loader import ModelLoader
from src.api.model_loader import THRESH_P
from src.api.monitoring import StatsTracker
from src.api.sql_validator import validate_query, ValidationError as SQLValidationError
from src.api.schemas import (
    CardGraphResponse, ExplainRequest, ExplainResponse, DualExplainResponse,
    HealthResponse, PredictRequest, PredictResponse,
    RingsResponse, FraudRing, StatsResponse,
    EventRecord, EventHistoryResponse,
    FlaggedTransaction, FlaggedResponse,
    MCPSlackRequest, MCPSlackResponse,
    SimilarTransaction, SimilarityResponse,
)

logger = logging.getLogger("atlasx")

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── Singletons ────────────────────────────────────────────────────────────────
loader  = ModelLoader()
tracker = StatsTracker()

# ── Postgres connection pool (async, for event sourcing + flagged queue) ──────
_pg_pool = None   # asyncpg.Pool – initialised in startup


async def _get_pg_pool():
    """Return the shared asyncpg connection pool, creating it on first call."""
    global _pg_pool
    if _pg_pool is None:
        try:
            import asyncpg
            dsn = os.getenv(
                "POSTGRES_DSN",
                "postgresql://postgres:postgres@localhost:5432/atlasx",
            )
            _pg_pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
            logger.info("[DB] asyncpg pool created.")
        except Exception as exc:
            logger.warning(f"[DB] PostgreSQL unavailable: {exc}")
            _pg_pool = None
    return _pg_pool

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ATLAS-X Fraud Detection API",
    version="3.0.0",
    description=(
        "XGBoost v4 + Neo4j graph layer + Event Sourcing + "
        "Dual LLM Explanations + pgvector Similarity + MCP Integration"
    ),
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
async def _startup() -> None:
    loader.load_all()
    # Initialise Postgres pool + schema
    pool = await _get_pg_pool()
    if pool:
        try:
            schema_path = Path(__file__).resolve().parents[2] / "src/monitoring/sql/schema.sql"
            schema_sql  = schema_path.read_text()
            async with pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("[DB] Schema initialised.")
            # Attempt pgvector setup (Feature 6)
            await _init_pgvector(pool)
        except Exception as exc:
            logger.warning(f"[DB] Schema init warning: {exc}")


async def _init_pgvector(pool) -> None:
    """
    Feature 6: Install pgvector extension and create transaction_embeddings table.
    Gracefully skips if pgvector is not installed in PostgreSQL.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS transaction_embeddings (
                    transaction_id TEXT PRIMARY KEY,
                    embedding      vector(495),
                    is_fraud       BOOLEAN DEFAULT FALSE,
                    created_at     TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            # HNSW index for fast cosine similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_txn_embeddings_hnsw
                ON transaction_embeddings
                USING hnsw (embedding vector_cosine_ops);
            """)
        logger.info("[pgvector] Extension and table ready.")
    except Exception as exc:
        logger.warning(f"[pgvector] Not available (skipping): {exc}")


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


def _normalise_vector(raw) -> list:
    """L2-normalise a feature vector for cosine similarity search in pgvector.
    Categorical columns are encoded as 0.0 (they don't contribute to similarity)."""
    vals = []
    for v in raw:
        try:
            vals.append(float(v) if v is not None else 0.0)
        except (TypeError, ValueError):
            vals.append(0.0)  # categorical → 0
    v = np.array(vals, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


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
    ts         = datetime.now(timezone.utc).isoformat()

    tracker.record(
        decision=decision,
        latency_ms=latency_ms,
        fraud_probability=fraud_prob,
        extra={
            "transaction_id":   transaction_id,
            "graph_risk_score": round(graph_risk, 4),
            "customer_segment": segment,
            "confidence":       confidence,
            "timestamp":        ts,
        },
    )

    result = {
        "transaction_id":    transaction_id,
        "fraud_probability": round(fraud_prob, 4),
        "graph_risk_score":  round(graph_risk, 4),
        "customer_segment":  segment,
        "decision":          decision,
        "confidence":        confidence,
        "latency_ms":        round(latency_ms, 2),
        "timestamp":         ts,
        "transaction_amt":   float(features.get("TransactionAmt") or 0.0),
        "_ring_details":     ring_details,
        # Feature 6: normalised feature vector for pgvector (stripped before API response)
        "_feature_vector":   _normalise_vector(X.values[0]),
    }

    return result


async def _persist_prediction(result: dict) -> None:
    """
    Feature 1: Write prediction to PostgreSQL predictions table and
    append a 'predicted' event to transaction_events for audit trail.
    Feature 6: Also store normalised feature vector in transaction_embeddings
    for pgvector HNSW similarity search.
    """
    pool = await _get_pg_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions
                    (transaction_id, fraud_probability, graph_risk_score,
                     decision, customer_segment, latency_ms, transaction_amt, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (transaction_id) DO UPDATE
                    SET fraud_probability = EXCLUDED.fraud_probability,
                        graph_risk_score  = EXCLUDED.graph_risk_score,
                        decision          = EXCLUDED.decision,
                        customer_segment  = EXCLUDED.customer_segment,
                        latency_ms        = EXCLUDED.latency_ms,
                        transaction_amt   = COALESCE(EXCLUDED.transaction_amt, predictions.transaction_amt)
                """,
                result["transaction_id"],
                result["fraud_probability"],
                result["graph_risk_score"],
                result["decision"],
                result["customer_segment"],
                result["latency_ms"],
                result.get("transaction_amt"),
            )
            event_data = {
                k: v for k, v in result.items()
                if k not in ("_ring_details", "_feature_vector")
            }
            await conn.execute(
                """
                INSERT INTO transaction_events
                    (transaction_id, event_type, data, created_at)
                VALUES ($1, $2, $3::jsonb, NOW())
                """,
                result["transaction_id"],
                "predicted",
                json.dumps(event_data),
            )
            # Feature 6: store embedding for pgvector similarity search
            # Use ground truth is_fraud_label if available, else fall back to decision
            fv = result.get("_feature_vector")
            if fv is not None:
                # Prefer ground truth label (set by Kafka consumer from dataset)
                # Fall back to BLOCK decision as a proxy
                gt_label = result.get("is_fraud_label")
                if gt_label is not None:
                    is_fraud = bool(gt_label)
                else:
                    is_fraud = result.get("decision") == "BLOCK"
                try:
                    await conn.execute(
                        """
                        INSERT INTO transaction_embeddings
                            (transaction_id, embedding, is_fraud, created_at)
                        VALUES ($1, $2::vector, $3, NOW())
                        ON CONFLICT (transaction_id) DO UPDATE
                            SET is_fraud = EXCLUDED.is_fraud
                        """,
                        result["transaction_id"],
                        "[" + ",".join(str(round(float(v), 6)) for v in fv) + "]",
                        is_fraud,
                    )
                except Exception as emb_exc:
                    logger.warning(f"[pgvector] Embedding insert failed for {result.get('transaction_id')}: {emb_exc}")
    except Exception as exc:
        logger.warning(f"[DB] Failed to persist prediction {result.get('transaction_id')}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  API v1 endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    result = _predict_core(req.transaction_id, req.card_id, req.features)
    if req.is_fraud_label is not None:
        result["is_fraud_label"] = req.is_fraud_label
    # Only store embedding when full features are present (Kafka stream)
    if not req.store_embedding:
        result.pop("_feature_vector", None)
    background_tasks.add_task(_persist_prediction, result)
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


# ── Feature 1: Event-Sourced Memory ──────────────────────────────────────────

@app.get("/api/v1/events/{transaction_id}", response_model=EventHistoryResponse)
async def get_events(transaction_id: str):
    """
    Feature 1 – Event-Sourced Memory / Audit Trail.

    Returns the complete chronological event history for a transaction from
    PostgreSQL.  Enables time-travel debugging and compliance auditing.

    Events are appended automatically on every predict call.  Additional
    event types (e.g. 'reviewed', 'escalated') can be written by downstream
    systems.
    """
    pool = await _get_pg_pool()
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL unavailable – event history not accessible.",
        )

    # SQL validator: ensure the query is safe before execution
    _safe_sql = (
        "SELECT event_type, data, created_at "
        "FROM transaction_events "
        "WHERE transaction_id = $1 "
        "ORDER BY created_at ASC"
    )
    try:
        validate_query(
            "SELECT event_type, data, created_at FROM transaction_events "
            "WHERE transaction_id = 'x' ORDER BY created_at ASC"
        )
    except SQLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(_safe_sql, transaction_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    events: List[EventRecord] = []
    for row in rows:
        raw_data = row["data"]
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)
        events.append(EventRecord(
            event_type=row["event_type"],
            data=raw_data,
            timestamp=row["created_at"].isoformat(),
        ))

    return EventHistoryResponse(
        transaction_id=transaction_id,
        events=events,
        total=len(events),
    )


# ── Feature 3: Human-in-the-Loop Checkpoints ─────────────────────────────────

@app.get("/api/v1/flagged", response_model=FlaggedResponse)
async def get_flagged_transactions(
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    Feature 3 – Human-in-the-Loop Review Queue.

    Returns all transactions where decision == 'FLAG', sorted by
    fraud_probability DESC (highest risk first).  Includes time waiting
    for analyst review.

    Fraud analysts use this endpoint to triage flagged transactions before
    they are automatically escalated or approved.
    """
    pool = await _get_pg_pool()
    if pool is None:
        # Fallback: return from in-memory tracker
        recent = tracker.recent()
        flagged_mem = [
            t for t in recent if t.get("decision") == "FLAG"
        ]
        flagged_mem.sort(key=lambda x: x.get("fraud_probability", 0), reverse=True)
        results = []
        now = datetime.now(timezone.utc)
        for t in flagged_mem[:limit]:
            ts_str = t.get("timestamp", now.isoformat())
            try:
                ts = datetime.fromisoformat(ts_str)
                minutes_waiting = (now - ts).total_seconds() / 60
            except Exception:
                minutes_waiting = 0.0
            results.append(FlaggedTransaction(
                transaction_id    =t.get("transaction_id", "unknown"),
                fraud_probability =t.get("fraud_probability", 0.0),
                graph_risk_score  =t.get("graph_risk_score", 0.0),
                customer_segment  =t.get("customer_segment", "Unknown"),
                decision          ="FLAG",
                latency_ms        =t.get("latency_ms", 0.0),
                created_at        =ts_str,
                minutes_waiting   =round(minutes_waiting, 1),
                flag_reason       =_flag_reason(t.get("fraud_probability", 0.0),
                                                t.get("graph_risk_score", 0.0)),
            ))
        return FlaggedResponse(flagged=results, total=len(results))

    # Validate the query pattern before execution (Feature 2)
    _safe_sql = """
        SELECT transaction_id, fraud_probability, graph_risk_score,
               customer_segment, decision, latency_ms, created_at
        FROM predictions
        WHERE decision = $1
        ORDER BY fraud_probability DESC
        LIMIT $2
    """
    try:
        validate_query(
            "SELECT transaction_id FROM predictions WHERE decision = 'FLAG'"
        )
    except SQLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(_safe_sql, "FLAG", limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    now = datetime.now(timezone.utc)
    results = []
    for row in rows:
        created_at = row["created_at"]
        minutes_waiting = (now - created_at).total_seconds() / 60
        results.append(FlaggedTransaction(
            transaction_id    =row["transaction_id"],
            fraud_probability =round(row["fraud_probability"] or 0.0, 4),
            graph_risk_score  =round(row["graph_risk_score"] or 0.0, 4),
            customer_segment  =row["customer_segment"] or "Unknown",
            decision          ="FLAG",
            latency_ms        =round(row["latency_ms"] or 0.0, 2),
            created_at        =created_at.isoformat(),
            minutes_waiting   =round(minutes_waiting, 1),
            flag_reason       =_flag_reason(
                row["fraud_probability"] or 0.0,
                row["graph_risk_score"] or 0.0,
            ),
        ))

    return FlaggedResponse(flagged=results, total=len(results))


def _flag_reason(fraud_prob: float, graph_risk: float) -> str:
    """Generate a human-readable reason for the FLAG decision."""
    reasons = []
    if fraud_prob >= 0.5:
        reasons.append(f"elevated fraud probability ({fraud_prob:.3f})")
    if graph_risk >= 0.5:
        reasons.append(f"high graph risk score ({graph_risk:.3f})")
    if not reasons:
        reasons.append("borderline model score with graph risk signal")
    return "; ".join(reasons).capitalize()


# ── Feature 4: MCP Server Integration ────────────────────────────────────────

@app.post("/api/v1/mcp/slack/alert", response_model=MCPSlackResponse)
async def mcp_slack_alert(req: MCPSlackRequest):
    """
    Feature 4 – MCP Enterprise Integration (Mock Implementation).

    Demonstrates the Model Context Protocol (MCP) pattern for enterprise
    integrations.  This mock implementation returns a success response
    without actually sending to Slack.

    Production integration:
    ─────────────────────────────────────────────────────────────────────
    Replace the mock body with an MCP client call:

        from mcp import ClientSession
        async with ClientSession(slack_mcp_server_url) as session:
            result = await session.call_tool(
                "slack_post_message",
                arguments={
                    "channel": req.channel,
                    "text":    req.message,
                    "blocks":  _build_fraud_alert_blocks(req),
                }
            )

    The same pattern applies to:
      - Google Drive (audit report upload)
      - Salesforce (case creation for high-risk customers)
      - PagerDuty (on-call escalation for BLOCK decisions)
      - Jira (ticket creation for analyst review)

    MCP servers are configured in .kiro/settings/mcp.json.
    ─────────────────────────────────────────────────────────────────────
    """
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"[MCP/Slack] Mock alert: txn={req.transaction_id} "
        f"channel={req.channel} msg={req.message[:80]}"
    )
    return MCPSlackResponse(
        status        ="sent",
        mock          =True,
        channel       =req.channel,
        transaction_id=req.transaction_id,
        message       =req.message,
        timestamp     =ts,
    )


# ── Feature 6: pgvector Similarity Search ────────────────────────────────────

@app.get("/api/v1/similar/{transaction_id}", response_model=SimilarityResponse)
async def similar_transactions(
    transaction_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
):
    """
    Feature 6 – pgvector HNSW Similarity Search.

    Finds the *top_k* most similar transactions to the given one using
    cosine similarity on the 498-dimensional feature embedding stored in
    PostgreSQL with the pgvector extension.

    Use case: "Show me past frauds that look like this transaction."

    The embedding is the normalised feature vector from the XGBoost model.
    Embeddings are stored when transactions are scored via /predict.

    Requires pgvector extension in PostgreSQL.
    """
    pool = await _get_pg_pool()
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL unavailable – similarity search not accessible.",
        )

    # Validate query pattern (Feature 2)
    try:
        validate_query(
            "SELECT transaction_id FROM transaction_embeddings WHERE transaction_id = 'x'"
        )
    except SQLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        async with pool.acquire() as conn:
            # Fetch the query transaction's embedding
            row = await conn.fetchrow(
                "SELECT embedding FROM transaction_embeddings WHERE transaction_id = $1",
                transaction_id,
            )
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No embedding found for transaction '{transaction_id}'. "
                        "Embeddings are stored when transactions are scored via /predict. "
                        "Ensure pgvector is installed and the transaction has been scored."
                    ),
                )

            # HNSW cosine similarity search
            # 1 - (embedding <=> query) gives cosine similarity
            similar_rows = await conn.fetch(
                """
                SELECT transaction_id,
                       1 - (embedding <=> $1) AS similarity_score,
                       is_fraud,
                       created_at
                FROM transaction_embeddings
                WHERE transaction_id != $2
                ORDER BY embedding <=> $1
                LIMIT $3
                """,
                row["embedding"],
                transaction_id,
                top_k,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"pgvector query failed: {exc}. "
                "Ensure pgvector extension is installed: "
                "CREATE EXTENSION IF NOT EXISTS vector;"
            ),
        )

    similar = [
        SimilarTransaction(
            transaction_id   =r["transaction_id"],
            similarity_score =round(float(r["similarity_score"]), 4),
            is_fraud         =bool(r["is_fraud"]),
            created_at       =r["created_at"].isoformat(),
        )
        for r in similar_rows
    ]

    return SimilarityResponse(
        query_transaction_id=transaction_id,
        similar_transactions=similar,
        total=len(similar),
    )


@app.post("/api/v1/explain")
async def explain(
    req: ExplainRequest,
    model: str = Query(default="qwen", description="LLM to use: 'qwen' | 'openai'"),
    compare: bool = Query(default=False, description="Return both Qwen and OpenAI explanations side-by-side"),
):
    """
    Feature 7 – Dual LLM Explanation System.

    Pass fraud_probability_override + decision_override to skip re-scoring
    and use the known values directly in the LLM prompt (correct behaviour
    when the frontend already has the result from /predict).
    """
    t_llm_start = time.perf_counter()

    has_overrides = (
        req.fraud_probability_override is not None
        and req.decision_override is not None
    )

    if has_overrides:
        # ── Use the values the frontend already knows for LLM prompt ──────────
        fp             = float(req.fraud_probability_override)
        decision       = req.decision_override
        segment        = req.customer_segment_override or "Regular"
        graph_risk_val = float(req.graph_risk_override or 0.0)
        threshold      = loader.thresholds.get(segment, 0.5)
        reasoning = (
            f"fraud_prob={fp:.3f} vs threshold={threshold:.3f} "
            f"(segment={segment}) → {decision}"
        )

        # Still run SHAP if features were provided (even partial)
        if req.features:
            try:
                import shap as _shap
                X         = loader.prepare_features(req.features)
                explainer = loader.get_shap_explainer()
                shap_vals = explainer(X)
                vals      = shap_vals[0].values
                names     = list(X.columns)
                top_idx   = np.argsort(np.abs(vals))[::-1][:10]
                top_shap: Dict[str, float] = {names[i]: round(float(vals[i]), 5) for i in top_idx}
            except Exception:
                top_shap = {}
        else:
            top_shap = {}

        # Check graph ring for explanation text
        graph_explanation = "No fraud ring connection detected."
        if loader.ring_checker is not None and req.features:
            try:
                ring = loader.ring_checker.check(req.features.get("card_id", "unknown"))
                if ring.get("in_fraud_ring"):
                    graph_explanation = (
                        f"Card shares device with "
                        f"{ring.get('device_frauds', 0)} confirmed fraud cards."
                    )
                    reasoning += f";  {graph_explanation}"
            except Exception:
                pass

    else:
        # ── Full re-score path (requires features) ────────────────────────────
        if req.features is None:
            raise HTTPException(
                status_code=422,
                detail="Provide either 'features' for re-scoring, or "
                       "'fraud_probability_override' + 'decision_override' "
                       "to use known values.",
            )

        X         = loader.prepare_features(req.features)
        explainer = loader.get_shap_explainer()

        import shap as _shap
        shap_vals = explainer(X)
        vals      = shap_vals[0].values
        names     = list(X.columns)
        top_idx   = np.argsort(np.abs(vals))[::-1][:10]
        top_shap  = {names[i]: round(float(vals[i]), 5) for i in top_idx}

        ring_info = ""
        if loader.ring_checker is not None:
            try:
                ring = loader.ring_checker.check(req.features.get("card_id", "unknown"))
                if ring.get("in_fraud_ring"):
                    ring_info = (
                        f"Card shares device with "
                        f"{ring.get('device_frauds', 0)} confirmed fraud cards."
                    )
            except Exception:
                pass
        graph_explanation = ring_info or "No fraud ring connection detected."

        fp             = float(loader.model.predict_proba(X)[:, 1][0])
        segment        = loader.segment_for(req.features)
        threshold      = loader.thresholds.get(segment, 0.5)
        decision       = _make_decision(fp, threshold, 0.0, bool(ring_info))
        graph_risk_val = float(req.features.get("graph_risk_score") or 0.0)
        reasoning = (
            f"fraud_prob={fp:.3f} vs threshold={threshold:.3f} "
            f"(segment={segment}) → {decision}"
        )
        if ring_info:
            reasoning += f";  {ring_info}"

    # ── LLM explanation(s) ────────────────────────────────────────────────────
    use_qwen   = compare or model == "qwen"
    use_openai = compare or model == "openai"

    qwen_explanation:   Optional[str] = None
    openai_explanation: Optional[str] = None

    llm_prompt = _build_llm_prompt(
        transaction_id=req.transaction_id,
        decision=decision,
        fraud_prob=fp,
        segment=segment,
        graph_risk=graph_risk_val,
        graph_explanation=graph_explanation,
        top_shap=top_shap,
    )

    if use_qwen:
        qwen_explanation = await _call_qwen(llm_prompt)
    if use_openai:
        openai_explanation = await _call_openai(llm_prompt)

    generation_time_ms = (time.perf_counter() - t_llm_start) * 1000
    model_used = "both" if compare else model

    return DualExplainResponse(
        transaction_id     =req.transaction_id,
        shap_values        =top_shap,
        graph_explanation  =graph_explanation,
        decision_reasoning =reasoning,
        qwen_explanation   =qwen_explanation,
        openai_explanation =openai_explanation,
        model_used         =model_used,
        generation_time_ms =round(generation_time_ms, 2),
    )


def _build_llm_prompt(
    *,
    transaction_id: str,
    decision: str,
    fraud_prob: float,
    segment: str,
    graph_risk: float,
    graph_explanation: str,
    top_shap: Dict[str, float],
) -> str:
    # Top 3 features only — keeps prompt short for faster Qwen response
    top_features = sorted(top_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    feature_lines = ", ".join(f"{name}={val:+.3f}" for name, val in top_features) if top_features else "N/A"

    # Accurate graph risk description
    if graph_risk >= 0.7:
        graph_desc = f"HIGH graph risk ({graph_risk:.3f}) — strong fraud network signal"
    elif graph_risk >= 0.4:
        graph_desc = f"ELEVATED graph risk ({graph_risk:.3f}) — connected to suspicious network"
    elif graph_risk > 0:
        graph_desc = f"LOW graph risk ({graph_risk:.3f})"
    else:
        graph_desc = "No graph risk signal"

    return (
        f"Fraud detection result for transaction {transaction_id}:\n"
        f"Decision: {decision} | FraudProb: {fraud_prob:.3f} | Segment: {segment}\n"
        f"Graph: {graph_desc}. {graph_explanation}\n"
        f"Top SHAP features: {feature_lines}\n\n"
        f"Write 2-3 sentences for a fraud analyst. "
        f"State the decision and probability first. "
        f"Explain the graph risk accurately. "
        f"Be specific, not generic."
    )


async def _call_qwen(prompt: str) -> str:
    """
    Feature 7: Call Ollama qwen2.5:7b (free, local).
    Timeout: 120s (model runs on CPU, typically 15-30s per response).
    Falls back to a canned message on failure.
    """
    import httpx

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model":  "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 200,   # cap tokens for speed
                        "temperature": 0.3,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except httpx.TimeoutException:
        return "Qwen explanation timed out (>120s). The model may be under load — try again."
    except Exception as exc:
        logger.warning(f"[Qwen] Call failed: {exc}")
        return f"Qwen unavailable: {exc}. Ensure Ollama is running: ollama serve"


async def _call_openai(prompt: str) -> str:
    """
    Feature 7: Call GPT-4o-mini via OpenAI API.
    Handles RateLimitError by returning a cached/Qwen fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI explanation unavailable: OPENAI_API_KEY not configured."

    try:
        from openai import AsyncOpenAI, RateLimitError
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            timeout=30.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        # Check for rate limit
        exc_name = type(exc).__name__
        if "RateLimit" in exc_name:
            logger.warning("[OpenAI] Rate limit hit – falling back to Qwen.")
            return await _call_qwen(prompt)
        logger.warning(f"[OpenAI] Call failed: {exc}")
        return f"OpenAI unavailable: {exc_name}."


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


# ── /api/v1/explain/llm (legacy – kept for backward compat) ──────────────────

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
async def explain_llm(req: LLMExplainRequest):
    """
    Natural-language explanation of a fraud decision.

    Default: Ollama qwen2.5:7b (free, local).
    Falls back to GPT-4o-mini if OPENAI_API_KEY is set and Qwen is unavailable.

    For the full dual-LLM comparison, use POST /api/v1/explain?compare=true.
    """
    prompt = _build_llm_prompt(
        transaction_id    =req.transaction_id,
        decision          =req.decision,
        fraud_prob        =req.fraud_probability,
        segment           =req.customer_segment,
        graph_risk        =req.graph_risk_score,
        graph_explanation =req.graph_explanation,
        top_shap          =req.shap_values,
    )
    explanation = await _call_qwen(prompt)
    # If Qwen failed, try OpenAI
    if explanation.startswith("Qwen unavailable") or explanation.startswith("Qwen explanation timed out"):
        openai_result = await _call_openai(prompt)
        if not openai_result.startswith("OpenAI unavailable"):
            explanation = openai_result

    return LLMExplainResponse(
        transaction_id=req.transaction_id,
        explanation=explanation,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold Optimization endpoints
# ══════════════════════════════════════════════════════════════════════════════

# In-memory store for the current override threshold (None = use thresholds.json)
_threshold_override: Optional[float] = None


@app.get("/api/v1/threshold/current")
async def get_current_threshold():
    """Return the currently active thresholds (override or defaults from thresholds.json)."""
    defaults = dict(loader.thresholds)
    override = _threshold_override
    return {
        "override_active": override is not None,
        "override_value":  override,
        "defaults":        defaults,
        "effective": {
            seg: override if override is not None else defaults.get(seg, 0.5)
            for seg in ["VIP", "Regular", "New"]
        },
    }


@app.post("/api/v1/threshold/apply")
async def apply_threshold(
    threshold: float = Query(..., ge=0.01, le=0.99, description="New global threshold (0.01–0.99)"),
):
    """
    Apply a new global threshold to all segments immediately.
    The change takes effect on the next /predict call — no restart needed.
    Pass threshold=null (or call /threshold/reset) to revert to per-segment defaults.
    """
    global _threshold_override
    _threshold_override = round(threshold, 4)
    # Patch the loader's thresholds in-place so _predict_core picks it up
    for seg in loader.thresholds:
        loader.thresholds[seg] = _threshold_override
    logger.info(f"[Threshold] Applied global override: {_threshold_override}")
    return {
        "status":    "applied",
        "threshold": _threshold_override,
        "message":   f"All segments now use threshold {_threshold_override:.4f}. "
                     f"Takes effect immediately on next /predict call.",
    }


@app.post("/api/v1/threshold/reset")
async def reset_threshold():
    """Revert to the original per-segment thresholds from thresholds.json."""
    global _threshold_override
    _threshold_override = None
    # Reload from artifact
    import json as _json
    thresh = _json.loads(THRESH_P.read_text())
    for seg in ["VIP", "Regular", "New"]:
        loader.thresholds[seg] = float(thresh["segments"][seg]["threshold"])
    logger.info("[Threshold] Reset to per-segment defaults from thresholds.json")
    return {
        "status":   "reset",
        "thresholds": dict(loader.thresholds),
        "message":  "Reverted to per-segment defaults (VIP=0.72, Regular=0.88, New=0.82).",
    }


@app.get("/api/v1/threshold/optimize")
async def optimize_threshold():
    """
    Scan all thresholds from 0.05 to 0.95 at 0.01 granularity and find the one
    that maximises net financial impact using real TransactionAmt from the dataset.

    Financial model:
      saved  = TP_amount × 4.60  (fraud cost multiplier, Risk Solutions 2024)
      lost   = FN_amount × 4.60
      fp_cost= FP_amount × 1.63  (revenue + churn, Aite-Novarica 2024)
      human  = flagged × 0.014 × $16.33  (analyst review cost)
      net    = saved - lost - fp_cost - human
    """
    pool = await _get_pg_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="PostgreSQL unavailable.")

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT fraud_probability, is_fraud_label, transaction_amt
                FROM predictions
                WHERE is_fraud_label IS NOT NULL AND transaction_amt IS NOT NULL
                """
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No labeled predictions with transaction amounts found. "
                   "Run the Kafka producer to populate data.",
        )

    import numpy as np

    probs  = np.array([float(r["fraud_probability"]) for r in rows])
    labels = np.array([bool(r["is_fraud_label"]) for r in rows])
    amts   = np.array([float(r["transaction_amt"]) for r in rows])

    FRAUD_MULT = 4.60
    FP_MULT    = 1.63
    COST_FLAG  = 16.33
    FLAG_RATE  = 0.014

    best_net = -1e12
    best_t   = 0.5
    curve: list[dict] = []

    for t_int in range(5, 96, 1):
        t = t_int / 100
        preds = probs >= t

        tp_amt = float(amts[preds & labels].sum())
        fn_amt = float(amts[~preds & labels].sum())
        fp_amt = float(amts[preds & ~labels].sum())
        tp_cnt = int((preds & labels).sum())
        fp_cnt = int((preds & ~labels).sum())
        fn_cnt = int((~preds & labels).sum())

        saved   = tp_amt * FRAUD_MULT
        lost    = fn_amt * FRAUD_MULT
        fp_cost = fp_amt * FP_MULT
        human   = int(preds.sum()) * FLAG_RATE * COST_FLAG
        net     = saved - lost - fp_cost - human

        prec = round(tp_cnt / max(1, tp_cnt + fp_cnt), 4)
        rec  = round(tp_cnt / max(1, tp_cnt + fn_cnt), 4)

        curve.append({
            "threshold":   t,
            "net_impact":  round(net, 2),
            "tp_amt":      round(tp_amt, 2),
            "fn_amt":      round(fn_amt, 2),
            "fp_amt":      round(fp_amt, 2),
            "tp":          tp_cnt,
            "fp":          fp_cnt,
            "fn":          fn_cnt,
            "precision":   prec,
            "recall":      rec,
            "f1":          round(2 * prec * rec / max(0.001, prec + rec), 4),
        })

        if net > best_net:
            best_net = net
            best_t   = t

    # Current threshold net impact
    current_t = list(loader.thresholds.values())[0] if _threshold_override is None else _threshold_override
    current_entry = min(curve, key=lambda x: abs(x["threshold"] - current_t))

    return {
        "optimal_threshold": best_t,
        "optimal_net_impact": round(best_net, 2),
        "current_threshold":  current_t,
        "current_net_impact": current_entry["net_impact"],
        "improvement":        round(best_net - current_entry["net_impact"], 2),
        "n_evaluated":        len(rows),
        "curve":              curve,   # full sweep for the chart
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Ground Truth Evaluation endpoint
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/evaluation")
async def get_evaluation(
    limit: int = Query(default=0, description="Max rows to evaluate (0 = all labeled)"),
    segment: Optional[str] = Query(default=None, description="Filter by customer_segment"),
):
    """
    Compare model predictions against ground truth isFraud labels.

    Returns full ML evaluation metrics:
      - Confusion matrix (TP, FP, FN, TN)
      - Precision, Recall, F1, Accuracy
      - Fraud detection rate (recall on BLOCK decisions)
      - False positive rate
      - Threshold analysis (precision/recall at different fraud_probability cutoffs)
      - Per-segment breakdown (VIP / Regular / New)
      - Score distribution (histogram buckets)

    Requires predictions table to have is_fraud_label populated.
    The Kafka producer sends is_fraud ground truth with each message,
    and the consumer writes it to the predictions table.
    """
    pool = await _get_pg_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="PostgreSQL unavailable.")

    try:
        async with pool.acquire() as conn:
            # Build query
            where_clauses = ["is_fraud_label IS NOT NULL"]
            params: list = []
            if segment:
                params.append(segment)
                where_clauses.append(f"customer_segment = ${len(params)}")

            where_sql = " AND ".join(where_clauses)
            limit_sql = f"LIMIT ${len(params)+1}" if limit > 0 else ""
            if limit > 0:
                params.append(limit)

            rows = await conn.fetch(
                f"""
                SELECT fraud_probability, decision, is_fraud_label, customer_segment,
                       COALESCE(transaction_amt, 0.0) as transaction_amt
                FROM predictions
                WHERE {where_sql}
                ORDER BY created_at DESC
                {limit_sql}
                """,
                *params,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No labeled predictions found. Ensure the Kafka producer is running "
                   "(it sends is_fraud ground truth) and the consumer has processed messages.",
        )

    # ── Compute metrics ───────────────────────────────────────────────────────
    import numpy as np

    probs     = np.array([float(r["fraud_probability"]) for r in rows])
    labels    = np.array([bool(r["is_fraud_label"]) for r in rows])
    amounts   = np.array([float(r["transaction_amt"]) for r in rows])
    segments  = [r["customer_segment"] for r in rows]

    n = len(rows)
    n_fraud = int(labels.sum())
    n_legit = n - n_fraud

    # ── Use the CURRENT ACTIVE threshold (not the stored decision column) ─────
    # Changing the threshold in the optimizer instantly re-computes TP/FP/FN
    # without needing to re-score all transactions.
    if _threshold_override is not None:
        active_threshold = _threshold_override
    else:
        # Use average of per-segment defaults
        active_threshold = sum(loader.thresholds.values()) / max(1, len(loader.thresholds))

    preds_block         = probs >= active_threshold
    preds_flag_or_block = probs >= (active_threshold * 0.4)

    tp = int((preds_block & labels).sum())
    fp = int((preds_block & ~labels).sum())
    fn = int((~preds_block & labels).sum())
    tn = int((~preds_block & ~labels).sum())

    precision = round(tp / max(1, tp + fp), 4)
    recall    = round(tp / max(1, tp + fn), 4)
    f1        = round(2 * precision * recall / max(0.001, precision + recall), 4)
    accuracy  = round((tp + tn) / n, 4)
    fpr       = round(fp / max(1, fp + tn), 4)

    # ── Real dollar amounts from TransactionAmt ───────────────────────────────
    tp_mask = preds_block & labels
    fp_mask = preds_block & ~labels
    fn_mask = ~preds_block & labels

    real_tp_amt = float(amounts[tp_mask].sum())
    real_fp_amt = float(amounts[fp_mask].sum())
    real_fn_amt = float(amounts[fn_mask].sum())

    avg_fraud_amt = float(amounts[labels].mean()) if labels.sum() > 0 else 0.0
    avg_legit_amt = float(amounts[~labels].mean()) if (~labels).sum() > 0 else 0.0

    # ── AUC (trapezoidal) ─────────────────────────────────────────────────────
    try:
        from sklearn.metrics import roc_auc_score
        auc = round(float(roc_auc_score(labels, probs)), 4)
    except Exception:
        auc = None

    # ── Threshold analysis: precision/recall at 10 cutoffs ───────────────────
    thresholds_analysis = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        preds_t = probs >= t
        tp_t = int((preds_t & labels).sum())
        fp_t = int((preds_t & ~labels).sum())
        fn_t = int((~preds_t & labels).sum())
        prec_t = round(tp_t / max(1, tp_t + fp_t), 4)
        rec_t  = round(tp_t / max(1, tp_t + fn_t), 4)
        f1_t   = round(2 * prec_t * rec_t / max(0.001, prec_t + rec_t), 4)
        thresholds_analysis.append({
            "threshold": t,
            "precision": prec_t,
            "recall":    rec_t,
            "f1":        f1_t,
            "flagged":   int(preds_t.sum()),
            # Real dollar amounts at this threshold
            "real_tp_amt": round(float(amounts[preds_t & labels].sum()), 2),
            "real_fp_amt": round(float(amounts[preds_t & ~labels].sum()), 2),
            "real_fn_amt": round(float(amounts[~preds_t & labels].sum()), 2),
        })

    # ── Per-segment breakdown ─────────────────────────────────────────────────
    seg_metrics: dict = {}
    for seg in ["VIP", "Regular", "New"]:
        mask = np.array([s == seg for s in segments])
        if mask.sum() == 0:
            continue
        seg_probs  = probs[mask]
        seg_labels = labels[mask]
        seg_amounts = amounts[mask]
        seg_block  = preds_block[mask]
        seg_tp = int((seg_block & seg_labels).sum())
        seg_fp = int((seg_block & ~seg_labels).sum())
        seg_fn = int((~seg_block & seg_labels).sum())
        seg_tn = int((~seg_block & ~seg_labels).sum())
        seg_prec = round(seg_tp / max(1, seg_tp + seg_fp), 4)
        seg_rec  = round(seg_tp / max(1, seg_tp + seg_fn), 4)
        seg_f1   = round(2 * seg_prec * seg_rec / max(0.001, seg_prec + seg_rec), 4)
        seg_metrics[seg] = {
            "n": int(mask.sum()),
            "n_fraud": int(seg_labels.sum()),
            "precision": seg_prec,
            "recall": seg_rec,
            "f1": seg_f1,
            "tp": seg_tp, "fp": seg_fp, "fn": seg_fn, "tn": seg_tn,
            # Real dollar amounts
            "real_tp_amt": round(float(seg_amounts[seg_block & seg_labels].sum()), 2),
            "real_fp_amt": round(float(seg_amounts[seg_block & ~seg_labels].sum()), 2),
            "real_fn_amt": round(float(seg_amounts[~seg_block & seg_labels].sum()), 2),
        }

    # ── Score distribution (10 buckets) ──────────────────────────────────────
    buckets = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (probs >= lo) & (probs < hi)
        buckets.append({
            "range": f"{lo:.1f}–{hi:.1f}",
            "total": int(mask.sum()),
            "fraud": int((mask & labels).sum()),
            "legit": int((mask & ~labels).sum()),
        })

    return {
        "summary": {
            "total_evaluated": n,
            "actual_fraud":    n_fraud,
            "actual_legit":    n_legit,
            "fraud_rate":      round(n_fraud / max(1, n), 4),
            "avg_fraud_amt":   round(avg_fraud_amt, 2),
            "avg_legit_amt":   round(avg_legit_amt, 2),
        },
        "active_threshold":  round(active_threshold, 4),
        "threshold_source":  "override" if _threshold_override is not None else "default",
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        # Real dollar amounts from actual TransactionAmt in the dataset
        "real_amounts": {
            "tp_amt": round(real_tp_amt, 2),   # actual fraud money caught (raw txn value)
            "fp_amt": round(real_fp_amt, 2),   # legit money wrongly blocked
            "fn_amt": round(real_fn_amt, 2),   # actual fraud money missed (raw txn value)
        },
        "metrics": {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "accuracy":  accuracy,
            "fpr":       fpr,
            "auc":       auc,
        },
        "per_segment":         seg_metrics,
        "threshold_analysis":  thresholds_analysis,
        "score_distribution":  buckets,
        "flag_count":          tp + fp,   # transactions above threshold (BLOCK decisions)
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy endpoints (backward-compat)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/predict")
async def api_predict_legacy(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    tx = payload.get("transaction", payload)
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
    background_tasks.add_task(_persist_prediction, result)
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
