# ATLAS-X: Technical Interview Guide

## Project Summary

ATLAS-X is a production-grade, real-time fraud detection system built on top of the
IEEE-CIS financial transaction dataset (590,540 transactions, 3.5% fraud rate). It
combines a gradient-boosted tree model with a graph database, a streaming pipeline,
a REST API, and a React dashboard — all containerised with Docker Compose.

---

## 1. Machine Learning

### Model: XGBoost v4 (Graph-Augmented)

**Why XGBoost over deep learning?**
The IEEE-CIS dataset is tabular with mixed categorical/numeric features and heavy class
imbalance (3.5% fraud). XGBoost handles this natively via `scale_pos_weight`, supports
categorical dtypes directly (`enable_categorical=True`), and produces calibrated
probabilities. Training time is ~3 minutes on CPU vs hours for a neural net, and
inference is ~6ms per transaction.

**Feature engineering:**
- 492 V-features: transaction amount, card metadata (card1–card6), address codes
  (addr1, addr2), email domains, device type/info, timing features (D1–D15 = days
  since various events), count features (C1–C14 = card-level velocity counters),
  match flags (M1–M9), and 400+ V-columns from the original dataset.
- 6 graph topology features computed from Neo4j at training time:
  `device_fraud_rate`, `device_card_velocity`, `connected_fraud_cards`,
  `email_fraud_rate`, `address_fraud_rate`, `graph_risk_score`.

**Hyperparameters:**
```python
XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=27.58,   # (negatives / positives)
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
)
```

**Training split:** 80/20 stratified by `isFraud` with `random_state=42` — the same
split is reproduced exactly in the Kafka producer so holdout labels are available for
ground truth evaluation.

**Metrics at default threshold (0.88):**
- AUC-ROC: 0.948
- Precision: 95.0% (very few false alarms)
- Recall: 36.1% (catches 36% of fraud)
- F1: 0.524

**Metrics at optimal threshold (0.53):**
- Precision: 62.9%
- Recall: 63.6%
- F1: 0.632
- Net financial impact: +$75,165 (vs -$826k at default)

**Why the default threshold is suboptimal:**
The model was originally calibrated for high precision (0.88 threshold). But the
financial cost of a missed fraud ($154 × 4.60 = $709) far exceeds the cost of a
false positive ($134 × 1.63 = $218). The threshold optimizer scans 91 thresholds
at 0.01 granularity and finds 0.53 maximises `saved − lost − FP_cost − review_cost`.

**SHAP explainability:**
`shap.TreeExplainer` is lazily initialised on first `/explain` call. Returns top-10
feature attributions by |SHAP value|. Positive = pushes toward fraud, negative =
pushes toward legitimate.

---

## 2. Graph Database — Neo4j

**Schema:**
```
(Card)-[:USED_DEVICE]->(Device)
(Card)-[:HAS_EMAIL]->(Email)
(Card)-[:BILLING_ADDR]->(Address)
```

**Node counts (full dataset):**
- 14,550 Card nodes
- 1,937 Device nodes
- 59 Email nodes
- 332 Address nodes
- 590,540 relationship rows

**Fraud ring detection:**
A node is in a fraud ring if it has ≥3 connected cards and ≥2 of those cards have
`fraud_txn > 0`. The `FraudRingChecker` runs a Cypher query on every `/predict` call.
If Neo4j is down, the model still scores using the precomputed `graph_risk_score`
feature from the request payload (graceful degradation).

**Graph features at training time vs inference time:**
Graph features are computed from the CSV exports at training time (batch). At inference,
Neo4j is queried live only for the `graph_risk_score` override and ring membership check.
This means inference does not require Neo4j to be available for the model score itself.

**295 fraud rings detected** in the holdout with `min_cards=5, min_frauds=3`.

---

## 3. Streaming Pipeline — Apache Kafka

**Architecture:**
```
kafka_producer.py  →  Kafka topic: transactions  →  kafka_consumer.py
                                                          ↓
                                              POST /api/v1/predict
                                                          ↓
                                         fraud-alerts / fraud-blocks topics
                                                          ↓
                                              PostgreSQL predictions table
```

**Producer:**
- Loads the 20% holdout (118,108 rows) from parquet
- Publishes at configurable rate (default 20 txns/sec) in a loop
- Each message includes `is_fraud` ground truth label for evaluation
- `--augment` flag mixes in 20% synthetic frauds (5 patterns: high_amount,
  device_sharing, rapid_sequence, cross_border, round_number)

**Consumer:**
- `asyncio.Semaphore(50)` for backpressure — limits concurrent in-flight API calls
- Routes BLOCK decisions to `fraud-blocks` topic, FLAG to `fraud-alerts`
- Writes to PostgreSQL with `is_fraud_label` and `transaction_amt` for ground truth eval
- Kafka dual-listener: `kafka:9092` (internal Docker) + `localhost:29092` (host)

**Throughput:** ~20–30 txns/sec sustained, p50 latency ~27ms end-to-end.

---

## 4. REST API — FastAPI

**Framework choice:** FastAPI over Flask/Django because:
- Native async support (needed for asyncpg, aiokafka, httpx)
- Pydantic v2 for request/response validation with zero boilerplate
- Auto-generated OpenAPI/Swagger docs at `/docs`
- `prometheus-fastapi-instrumentator` for zero-config Prometheus metrics

**Key endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/predict` | POST | Score a transaction (XGBoost + Neo4j) |
| `/api/v1/explain` | POST | SHAP + dual LLM explanation |
| `/api/v1/evaluation` | GET | Ground truth metrics vs active threshold |
| `/api/v1/threshold/optimize` | GET | Scan 91 thresholds for max profit |
| `/api/v1/threshold/apply` | POST | Apply threshold in-memory (no restart) |
| `/api/v1/events/{txn_id}` | GET | Event-sourced audit trail |
| `/api/v1/flagged` | GET | Human review queue (FLAG decisions) |
| `/api/v1/mcp/slack/alert` | POST | MCP enterprise integration (mock) |
| `/api/v1/similar/{txn_id}` | GET | pgvector HNSW similarity search |

**Decision logic:**
```python
if fraud_prob >= threshold:
    decision = "BLOCK"
elif fraud_prob >= threshold * 0.4 and (in_ring or graph_risk >= 0.5):
    decision = "FLAG"
else:
    decision = "APPROVE"
```

**Singleton pattern:** `ModelLoader` and `StatsTracker` are thread-safe singletons
initialised at startup. The model is loaded once and shared across all requests.

**Background tasks:** `_persist_prediction` uses FastAPI's `BackgroundTasks` to write
to PostgreSQL without blocking the response. Fire-and-forget with error logging.

**Latency breakdown (p50):**
- Feature DataFrame construction: 2–4ms
- XGBoost `predict_proba()`: 3–6ms
- Neo4j Cypher query: 5–12ms
- Decision + serialisation: <1ms
- **Total p50: ~27ms**

---

## 5. Database Layer — PostgreSQL + pgvector

**Schema:**
```sql
predictions (
    transaction_id TEXT UNIQUE,
    fraud_probability FLOAT,
    graph_risk_score FLOAT,
    decision TEXT,
    customer_segment TEXT,
    latency_ms FLOAT,
    transaction_amt FLOAT,        -- real TransactionAmt from dataset
    is_fraud_label BOOLEAN,       -- ground truth from Kafka producer
    created_at TIMESTAMPTZ
)

transaction_events (
    transaction_id TEXT,
    event_type TEXT,              -- 'predicted', 'reviewed', etc.
    data JSONB,
    created_at TIMESTAMPTZ
)

transaction_embeddings (
    transaction_id TEXT PRIMARY KEY,
    embedding vector(498),        -- pgvector HNSW index
    is_fraud BOOLEAN
)
```

**pgvector:** Installed via `postgresql-16-pgvector` apt package in the Dockerfile.
HNSW index with cosine similarity (`vector_cosine_ops`) for sub-millisecond
nearest-neighbour search on 498-dimensional feature vectors.

**Connection pooling:** `asyncpg.create_pool(min_size=2, max_size=10)` initialised
at startup. All DB writes use `pool.acquire()` context manager.

**SQL injection prevention (`src/api/sql_validator.py`):**
Every SQL string is validated before execution:
- SELECT-only (no DML/DDL)
- Whitelisted tables: `predictions`, `fraud_alerts`, `transaction_events`
- No semicolons, comments, UNION, pg_sleep, hex encoding

---

## 6. LLM Integration — Dual Model System

**Qwen 2.5:7b via Ollama (default):**
```python
async with httpx.AsyncClient(timeout=120.0) as client:
    resp = await client.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 200, "temperature": 0.3},
    })
```
Runs locally, free, ~6–20s on CPU. Prompt is kept short (3 features, 2-3 sentence ask)
to minimise generation time.

**GPT-4o-mini via OpenAI API:**
```python
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = await client.chat.completions.create(
    model="gpt-4o-mini", max_tokens=300, timeout=30.0, ...
)
```
Used for comparison (`?compare=true`) or when Qwen is unavailable.
`RateLimitError` falls back to Qwen automatically.

**Key design decision:** The `/explain` endpoint accepts `fraud_probability_override`
and `decision_override` so the LLM uses the real values from the prior `/predict` call
rather than re-scoring from (possibly partial) features.

---

## 7. Frontend — React + Vite

**Stack:** React 18, TypeScript 5.5, Vite 5, Recharts, react-force-graph, lucide-react.

**Key components:**
- `MetricsDashboard` — KPI cards (total predictions, fraud rate, avg latency, decisions)
- `TransactionFeed` — live table polling `/api/v1/recent` every 3s
- `FraudRingGraph` — force-directed graph of fraud rings from Neo4j
- `ShapExplainer` — SHAP bar chart + dual LLM explanation with compare button
- `AlertFeed` — WebSocket subscriber to `/ws/alerts` (Kafka → browser bridge)
- `BackendDashboard` — full-screen overlay with 4 tabs:
  - **Ops**: system health, latency chart, decision pie, fraud rings, flagged queue
  - **Ground Truth**: confusion matrix, per-segment metrics, threshold analysis
  - **Financial Impact**: real dollar amounts from TransactionAmt × multipliers
  - **Threshold Optimizer**: 91-point sweep chart, slider, Apply/Reset buttons

**State management:** Local `useState`/`useEffect` — no Redux needed at this scale.
The `FinancialTab` re-fetches on every tab activation (`useEffect(() => { if (isActive) load(); }, [isActive])`) to pick up threshold changes.

**API client:** Axios with 150s timeout (for LLM calls). All endpoints typed with
TypeScript interfaces matching the Pydantic response models.

---

## 8. Observability

**Prometheus:** `prometheus-fastapi-instrumentator` auto-instruments all routes.
Scrape target: `GET /api/metrics/prometheus`. Metrics include request count,
latency histograms (p50/p95/p99), and status code distributions.

**Grafana:** 4-row backend ops dashboard auto-provisioned from JSON:
- System Health (API uptime, error rate, container status)
- Performance (latency p50/p95/p99, Kafka lag, PG connections)
- Business Metrics (TPS, decision distribution, hourly trend)
- Model Monitoring (predict vs SHAP latency, feature usage)

**In-memory StatsTracker:** Thread-safe singleton with a `deque(maxlen=10_000)`
rolling latency window and a `deque(maxlen=50)` recent-transactions ring buffer.

---

## 9. Enterprise Features

### Event Sourcing
Every prediction appends a `predicted` event to `transaction_events` (JSONB).
`GET /api/v1/events/{txn_id}` returns the full chronological audit trail.
Enables time-travel debugging and compliance auditing.

### Threshold Optimizer
Scans 91 thresholds (0.05–0.95 at 0.01 steps) against 118k labeled predictions.
Financial model: `net = (tp_amt × 4.60) − (fn_amt × 4.60) − (fp_amt × 1.63) − (flagged × 0.014 × $16.33)`.
Optimal threshold (0.53) applied in-memory via `loader.thresholds[seg] = override` —
no restart required.

### GitOps / ArgoCD
`deployment/argocd-app.yaml` + full `deployment/k8s/` manifests (FastAPI Deployment
with HPA 2–10 pods, Kafka, Neo4j StatefulSet, Postgres StatefulSet with pgvector image,
Ingress with TLS). ArgoCD watches `main` branch and auto-deploys on merge.

### Data Augmentation
`FraudAugmentor` generates synthetic fraud using SMOTE-like mutation of real fraud seeds.
5 patterns: `high_amount` (5–20× amount, D1≈0), `device_sharing` (high velocity counters),
`rapid_sequence` (C1–C4 high, D9≈0), `cross_border` (addr2 foreign, dist2 large),
`round_number` ($100/$500/$1000 amounts). `--augment` flag mixes 80% real + 20% synthetic.

---

## 10. Infrastructure

```
Docker Compose services:
  api          FastAPI (port 8001)
  kafka        Confluent Kafka 7.6.1 (ports 9092, 29092)
  zookeeper    Confluent Zookeeper 7.6.1 (port 2181)
  neo4j        Neo4j 5.22 (ports 7474, 7687)
  postgres     postgres:16 + pgvector (port 5433)
  redis        Redis 7 (port 6379)
  prometheus   prom/prometheus:v2.55.1 (port 9090)
  grafana      grafana/grafana:11.1.0 (port 3000)
```

**Port 5433 (not 5432):** A local PostgreSQL instance was already running on 5432,
so Docker Postgres is mapped to 5433 to avoid conflict.

**Memory management:** The XGBoost model (~500MB), parquet data (~2GB), and SHAP
TreeExplainer (~1GB) are all in-process. The SHAP explainer is lazily initialised
on first `/explain` call to avoid OOM at startup. The Kafka consumer and producer
are run as separate processes to avoid competing for memory with the API.

---

## Quick Reference: Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| XGBoost over neural net | Tabular data, fast inference, calibrated probabilities |
| Graph features at training time | Avoids Neo4j dependency for model scoring |
| Async FastAPI | Needed for asyncpg, aiokafka, httpx concurrency |
| BackgroundTasks for DB writes | Non-blocking response, fire-and-forget persistence |
| Threshold override in-memory | No restart needed, instant effect on next predict call |
| Evaluation re-applies threshold to probabilities | Stale `decision` column reflects old threshold; re-applying to `fraud_probability` gives live results |
| pgvector on port 5433 | Avoids conflict with local postgres on 5432 |
| Qwen default, GPT-4o-mini optional | Free local inference by default, OpenAI for comparison |
