# ATLAS-X System Architecture

---

## Data Flow

```
                          TRAINING PATH
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  data/raw/                                              │
  │  ├── train_transaction.csv   (590k rows, IEEE-CIS)      │
  │  ├── train_identity.csv                                 │
  │  └── neo4j/                                             │
  │      ├── cards.csv                                      │
  │      ├── devices.csv                                    │
  │      ├── emails.csv                                     │
  │      ├── addresses.csv                                  │
  │      └── relationships.csv                              │
  │             │                                           │
  │             ▼                                           │
  │  src/features/build_full_features.py                    │
  │  → 492 V-engineered features                            │
  │             │                                           │
  │             ▼                                           │
  │  src/features/engineer_graph_features.py                │
  │  → 6 graph topology features (device/email/addr)        │
  │             │                                           │
  │             ▼                                           │
  │  data/processed/train_with_graph_features.parquet       │
  │  (498 feature columns, 590k rows)                       │
  │             │                                           │
  │             ▼                                           │
  │  src/models/train_v4_with_graph.py                      │
  │  → XGBoost v4 model  (src/models/atlass_x_xgb_v4.pkl)  │
  │  → Per-segment thresholds (thresholds.json)             │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

                        INFERENCE PATH
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  HTTP POST /api/v1/predict                              │
  │             │                                           │
  │             ▼                                           │
  │  ModelLoader.prepare_features()                         │
  │  • Build DataFrame from features dict                   │
  │  • Cast categorical columns                             │
  │  • Fill missing values                                  │
  │             │                                           │
  │    ┌────────┴────────┐                                  │
  │    ▼                 ▼                                  │
  │  XGBoost           Neo4j                                │
  │  predict_proba()   FraudRingChecker.check(card_id)      │
  │    │                 │                                  │
  │    └────────┬────────┘                                  │
  │             ▼                                           │
  │  _make_decision(fraud_prob, threshold, graph_risk)      │
  │             │                                           │
  │    ┌────────┴─────────────────────┐                     │
  │    ▼           ▼                  ▼                     │
  │  APPROVE     FLAG              BLOCK                    │
  │             │                                           │
  │             ▼                                           │
  │  StatsTracker.record()  (in-memory ring buffer)         │
  │             │                                           │
  │             ▼                                           │
  │  PredictResponse (JSON)                                 │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

                        STREAMING PATH
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  kafka_producer.py                                      │
  │  • Loads holdout parquet                                │
  │  • Rate-controlled publish (default 200 txns/sec)       │
  │             │                                           │
  │             ▼                                           │
  │  Kafka topic: transactions  (localhost:29092)           │
  │             │                                           │
  │             ▼                                           │
  │  kafka_consumer.py (aiokafka)                           │
  │  • Semaphore(50) for backpressure                       │
  │  • POST /api/v1/predict per message                     │
  │             │                                           │
  │    ┌────────┴────────────────┐                          │
  │    ▼                         ▼                          │
  │  Kafka: fraud-alerts       Postgres: predictions        │
  │  Kafka: fraud-blocks       (all scored transactions)    │
  │    │                                                    │
  │    ▼                                                    │
  │  alert_consumer.py                                      │
  │  • Prints ⚠️/🛑 alerts to terminal                       │
  │  • Logs to Postgres: fraud_alerts                       │
  │    │                                                    │
  │    ▼                                                    │
  │  WS /ws/alerts  (Kafka → WebSocket bridge)              │
  │  • React AlertFeed panel subscribes                     │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

---

## Component Details

### XGBoost v4

| Property | Value |
|---|---|
| Algorithm | Gradient-boosted trees (XGBoost 2.x) |
| n_estimators | 600 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 20 (class imbalance) |
| Feature columns | 498 (492 V-features + 6 graph features) |
| Training rows | ~472k (80% stratified split) |
| Categorical handling | XGBoost native (enable_categorical=True) |

**Graph features computed at training time** from Neo4j CSVs — not at inference
time, so inference does not require a live Neo4j connection for model scoring.
Neo4j is only queried live for the `graph_risk_score` override and ring detection.

### Neo4j Graph

```
(Card)-[:USED_DEVICE]->(Device)
(Card)-[:HAS_EMAIL]->(Email)
(Card)-[:BILLING_ADDR]->(Address)
```

The `FraudRingChecker` runs a Cypher query against the live graph on each
`/predict` call. If Neo4j is down the model still scores using the precomputed
`graph_risk_score` feature from the request payload (defaults to 0.0 if absent).

Fraud ring detection logic: a node is in a ring if it has ≥ 3 connected cards
and ≥ 2 of those cards have `fraud_txn > 0`.

### FastAPI + ModelLoader

`ModelLoader` is a thread-safe singleton initialised at API startup:

1. Loads `atlass_x_xgb_v4_graph.pkl` (XGBoost booster)
2. Loads `thresholds.json` (per-segment thresholds)
3. Infers categorical column names from a parquet sample
4. Lazily initialises the SHAP `TreeExplainer` on first `/explain` call
5. Optionally connects to Neo4j via `FraudRingChecker`

The `StatsTracker` singleton keeps:
- Rolling latency deque (maxlen=10,000)
- Decision counters
- `_recent` ring buffer (maxlen=50) for `/api/v1/recent`

### Kafka Setup

| Topic | Producer | Consumer | Purpose |
|---|---|---|---|
| `transactions` | `kafka_producer.py` | `kafka_consumer.py` | Raw transaction feed |
| `fraud-alerts` | `kafka_consumer.py` | `alert_consumer.py`, WebSocket | FLAG decisions |
| `fraud-blocks` | `kafka_consumer.py` | `alert_consumer.py` | BLOCK decisions |

Dual listener configuration (docker-compose):
- `kafka:9092` — internal Docker network
- `localhost:29092` — external host access

### Postgres Schema

```sql
CREATE TABLE predictions (
    id               SERIAL PRIMARY KEY,
    transaction_id   TEXT UNIQUE NOT NULL,
    fraud_probability FLOAT,
    graph_risk_score  FLOAT,
    decision         TEXT,
    customer_segment TEXT,
    latency_ms       FLOAT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE fraud_alerts (
    id               SERIAL PRIMARY KEY,
    transaction_id   TEXT UNIQUE NOT NULL,
    fraud_probability FLOAT,
    graph_risk_score  FLOAT,
    decision         TEXT,
    customer_segment TEXT,
    is_fraud_label   BOOL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Latency Breakdown

Measured on Apple M-series CPU, Neo4j available locally.

| Step | Typical time |
|---|---|
| Feature DataFrame construction | 2–4 ms |
| XGBoost `predict_proba()` | 3–6 ms |
| Neo4j Cypher query (ring check) | 5–12 ms |
| Decision + confidence calculation | < 1 ms |
| JSON serialisation + HTTP overhead | 1–3 ms |
| **Total p50** | **~15 ms** |
| **Total p99** | **~45 ms** |

When Neo4j is disabled, p50 drops to ~8 ms.

---

## Decision Logic

```
fraud_prob = XGBoost.predict_proba(features)[1]
threshold  = thresholds[customer_segment]   # VIP=0.72, Regular=0.88, New=0.82

if fraud_prob >= threshold:
    decision = BLOCK
elif fraud_prob >= threshold * 0.4 AND (in_fraud_ring OR graph_risk >= 0.5):
    decision = FLAG
else:
    decision = APPROVE

confidence:
  BLOCK   → fraud_prob
  APPROVE → 1 - fraud_prob
  FLAG    → |fraud_prob - 0.5| + 0.5
```

---

## Scalability Notes

### Horizontal scaling

- The FastAPI app is stateless except for the in-memory `StatsTracker`. For
  multi-instance deployments, replace `StatsTracker` with a Redis-backed
  counter (the interface is unchanged — swap the implementation in
  `src/api/monitoring.py`).
- The XGBoost model is read-only after load; multiple Uvicorn workers share it
  safely with `--workers N`.

### Kafka throughput

The consumer's `asyncio.Semaphore(50)` limits concurrent in-flight API calls.
At 200 txns/sec input and 15 ms per inference, 3 concurrent workers suffice;
the semaphore provides headroom for latency spikes.

### Neo4j

The `_query_rings` endpoint creates a new driver connection per request. For
production, use a connection pool (`neo4j.AsyncGraphDatabase.driver` with
`max_connection_pool_size`).

### Model updates

To deploy a new model:
1. Train and save to `src/models/atlass_x_xgb_v5.pkl`
2. Update `MODEL_PATH` in `src/api/model_loader.py`
3. `kill -HUP $(cat uvicorn.pid)` — Uvicorn reloads without downtime

---

## Observability

| Signal | Endpoint | Tool |
|---|---|---|
| Request count, latency histograms | `GET /api/metrics/prometheus` | Prometheus |
| Fraud rate, decision distribution | `GET /api/v1/stats` | Grafana / custom |
| Per-transaction audit trail | Postgres `predictions` table | SQL / BI |
| Real-time alerts | `WS /ws/alerts` | React dashboard |
| System resources | `GET /api/metrics` (legacy) | psutil |
