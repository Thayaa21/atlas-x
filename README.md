# ATLAS-X: Real-Time Fraud Detection System

A production-grade fraud detection engine combining XGBoost v4 (graph-augmented) with
a Neo4j fraud ring graph, Kafka streaming pipeline, FastAPI inference service, and a
React real-time dashboard.

---

## Overview

ATLAS-X scores financial transactions in real time, combining gradient-boosted tree
inference with graph topology features extracted from a Neo4j card–device–email–address
network. Suspicious transactions are streamed via Kafka to a React dashboard and
optionally routed to Postgres for audit logging.

---

## Performance Metrics

| Model | AUC | Recall | Precision | F1 | Avg Latency |
|---|---|---|---|---|---|
| XGBoost v3 (baseline) | 0.9528 | 58.05% | 84.32% | 68.76% | — |
| **XGBoost v4 (graph-augmented)** | **0.9573** | **59.18%** | **84.75%** | **69.70%** | **~17 ms** |

**Per-segment thresholds (v4):**

| Segment | Threshold | Recall | Precision | F1 |
|---|---|---|---|---|
| VIP | 0.72 | 60.00% | 78.38% | 67.97% |
| Regular | 0.88 | 51.84% | 84.35% | 64.21% |
| New account | 0.82 | 66.24% | 84.79% | 74.37% |

Holdout: 118,108 transactions (4,133 fraud) from the IEEE-CIS dataset.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ATLAS-X System                               │
│                                                                     │
│   Raw Data            Feature Engineering         Model Layer       │
│  ──────────          ──────────────────────      ──────────────     │
│  IEEE-CIS CSV   ──►  V-features (390 cols)  ──►  XGBoost v4        │
│  Neo4j CSVs     ──►  Graph features (6 cols)     (598 cols total)   │
│                       device_fraud_rate                             │
│                       device_card_velocity      Neo4j Graph         │
│                       connected_fraud_cards ◄──  Card–Device        │
│                       email_fraud_rate           Card–Email         │
│                       address_fraud_rate         Card–Address       │
│                       graph_risk_score                              │
│                                                                     │
│   Inference Layer                Streaming Layer                    │
│  ──────────────────             ────────────────────────────        │
│  FastAPI /api/v1/predict  ──►   Kafka topic: transactions           │
│  3-tier decision logic:         Kafka consumer (aiokafka)           │
│    BLOCK  (prob ≥ threshold)    ├── FLAG  → fraud-alerts topic      │
│    FLAG   (ring + elevated)     ├── BLOCK → fraud-blocks topic      │
│    APPROVE (low risk)           └── All   → Postgres predictions    │
│                                                                     │
│   Observability                 Frontend                            │
│  ──────────────                ──────────────────────────────       │
│  Prometheus /metrics    ──►    React dashboard (Vite + Recharts)    │
│  Grafana dashboards             MetricsDashboard (pie + KPIs)       │
│  In-memory StatsTracker         TransactionFeed  (live table)       │
│                                 FraudRingGraph   (force graph)      │
│                                 ShapExplainer    (SHAP bar chart)   │
│                                 AlertFeed        (WS alerts)        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 20+

### 1. Install and configure

```bash
./setup.sh
```

This installs Python dependencies, starts Docker services (Kafka, Neo4j, Redis,
Postgres), initialises the Postgres schema, and installs frontend packages.

### 2. Start everything

```bash
./start.sh
```

Starts Docker services, FastAPI (port 8001), Kafka consumer/producer, and the React dashboard in the correct order. Each step waits for the previous to be healthy.

| Service | URL |
|---|---|
| React dashboard | http://localhost:5173 |
| FastAPI | http://localhost:8001 |
| Swagger docs | http://localhost:8001/docs |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Neo4j Browser | http://localhost:7474 |

### 3. Stop everything

```bash
./stop.sh
```

### 4. Run a quick demo

```bash
./demo.sh
```

Sends sample transactions and prints scored results.

### 5. Start components individually (optional)

```bash
# API only
./run_api.sh

# Kafka consumer + producer
python -m src.streaming.kafka_consumer
python -m src.streaming.kafka_producer --rate 200 --count 1000

# Frontend only
cd frontend && npm run dev
```

---

## Project Structure

```
atlas-x/
├── data/
│   ├── raw/                        # IEEE-CIS CSV files (not committed)
│   ├── processed/                  # Parquet feature files
│   └── neo4j/                      # Graph CSV files (cards, devices, emails, addresses)
├── deployment/
│   ├── argocd-app.yaml             # ArgoCD GitOps application
│   └── k8s/                        # Kubernetes manifests (API, Kafka, Neo4j, Postgres)
├── docker/
│   ├── docker-compose.yml          # Kafka, Neo4j, Redis, Postgres, Grafana, Prometheus
│   ├── prometheus/
│   └── grafana/                    # Dashboard provisioning
├── docs/
│   ├── API.md                      # endpoint reference
│   ├── ARCHITECTURE.md             # detailed architecture
│   ├── BUSINESS_OVERVIEW.md        # business context
│   └── TECHNICAL_INTERVIEW.md      # technical deep-dive
├── frontend/
│   ├── src/
│   │   ├── api/client.ts           # axios API wrapper
│   │   ├── components/
│   │   │   ├── AlertFeed.tsx       # WebSocket live alerts
│   │   │   ├── BackendDashboard.tsx # Grafana/ops metrics panel
│   │   │   ├── EvaluationPanel.tsx  # model evaluation metrics
│   │   │   ├── FinancialImpactPanel.tsx # financial impact view
│   │   │   ├── FraudRingGraph.tsx   # Neo4j force graph
│   │   │   ├── MetricsDashboard.tsx # KPIs + pie chart
│   │   │   ├── ShapExplainer.tsx    # SHAP bar chart + LLM explanation
│   │   │   ├── ThresholdBar.tsx     # threshold visualisation
│   │   │   ├── ThresholdOptimizer.tsx # per-segment threshold tuning
│   │   │   ├── TransactionFeed.tsx  # live transaction table
│   │   │   └── WebSocketMonitor.tsx # WS connection status
│   │   └── utils/mockData.ts       # offline mock data
│   ├── package.json
│   └── vite.config.ts
├── monitoring/
│   ├── README.md                   # Grafana setup instructions
│   └── backend-ops-dashboard.json  # Grafana dashboard definition
├── results/
│   ├── v4_graph_metrics.json
│   └── model_comparison.png
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app (11 v1 endpoints)
│   │   ├── model_loader.py         # singleton model + SHAP loader
│   │   ├── monitoring.py           # in-memory StatsTracker
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── sql_validator.py        # SQL injection prevention (Feature 2)
│   ├── evaluation/
│   │   └── compare_models.py       # v3 vs v4 comparison
│   ├── features/
│   │   ├── build_full_features.py  # V-features pipeline
│   │   └── engineer_graph_features.py  # 6 graph topology features
│   ├── graph/
│   │   ├── fraud_ring_checker.py   # live Neo4j card risk check
│   │   └── load_neo4j.py           # Neo4j CSV loader
│   ├── models/
│   │   ├── train_v3_model.py
│   │   └── train_v4_with_graph.py
│   ├── optimization/
│   │   └── threshold_optimizer.py  # per-segment threshold search
│   ├── rl/
│   │   ├── dqn_agent_v2.py         # 4-action DQN (contextual bandit)
│   │   └── fraud_env.py            # reward-shaping environment
│   └── streaming/
│       ├── kafka_producer.py
│       ├── kafka_consumer.py
│       ├── data_augmentor.py       # synthetic fraud generation (Feature 9)
│       ├── alert_consumer.py
│       ├── init_db.py
│       └── monitor.py
├── tests/
│   ├── test_api_endpoints.py       # API integration tests
│   ├── test_streaming.py           # end-to-end Kafka test
│   └── sample_transactions.json    # holdout samples for tests
├── README.md
├── requirements.txt
├── setup.sh
├── start.sh                        # start all services in order
├── stop.sh                         # stop all services cleanly
├── run_api.sh                      # start API only
└── demo.sh
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML model | XGBoost 2.x (GPU-optional), SHAP |
| Graph DB | Neo4j 5 (Bolt), neo4j-driver |
| API | FastAPI 0.111, Uvicorn, Pydantic v2 |
| Streaming | Apache Kafka 3.7 (Confluent), aiokafka |
| Storage | PostgreSQL 16 (asyncpg), Redis 7 |
| Observability | Prometheus, Grafana, prometheus-fastapi-instrumentator |
| Frontend | React 18, Vite 5, Recharts, react-force-graph, lucide-react |
| Containerisation | Docker Compose |
| Language | Python 3.11, TypeScript 5.5 |

---

## Model Details

### XGBoost v4 (graph-augmented)

- **Base features:** 492 V-engineered features (transaction amount, card, address,
  email, device, identity, C/D/M/V columns)
- **Graph features (6 new):**
  - `device_fraud_rate` — fraction of co-device cards that committed fraud
  - `device_card_velocity` — number of cards sharing the same device
  - `connected_fraud_cards` — total fraud cards reachable via device/email/address
  - `email_fraud_rate` — fraud rate among cards sharing the same email domain
  - `address_fraud_rate` — fraud rate among cards sharing the same billing address
  - `graph_risk_score` — composite score (0–1)
- **Hyperparameters:** n_estimators=600, max_depth=6, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=20
- **Training:** 80/20 stratified split of 590k IEEE-CIS transactions

### Segment-aware thresholds

Segments are assigned per transaction using recency (D1) and tenure (D15).
Separate thresholds minimise false-positives on VIP accounts while maintaining
high recall on new accounts.

### Neo4j graph

Nodes: `Card`, `Device`, `Email`, `Address`  
Relationships: `USED_DEVICE`, `HAS_EMAIL`, `BILLING_ADDR`  
Used for: fraud ring detection, live graph risk override, SHAP graph explanation.

---

## Testing

```bash
# API endpoint tests (requires API running on :8001)
python tests/test_api_endpoints.py

# SQL validator tests (no API required)
python tests/test_sql_validator.py

# MCP integration tests (requires API running on :8001)
python tests/test_mcp_integration.py

# Kafka streaming integration test (requires Kafka + Postgres)
python tests/test_streaming.py
```

Test suite: 8 API tests + 35 SQL validator tests + 6 MCP tests.

---

## Enterprise Features

### 1. Event-Sourced Memory (Audit Trail)

Every prediction is persisted to PostgreSQL with a full event history.

```bash
# Get complete audit trail for a transaction
curl http://localhost:8001/api/v1/events/txn-001
```

Returns chronological array of all events: `[{event_type, data, timestamp}, ...]`.
Enables time-travel debugging and compliance auditing.

### 2. SQL Injection Prevention

All PostgreSQL queries are pre-validated by `src/api/sql_validator.py` before execution:
- SELECT-only (no DML/DDL)
- Whitelisted tables: `predictions`, `fraud_alerts`, `transaction_events`
- No injection patterns: semicolons, comments, UNION, pg_sleep, hex encoding

```python
from src.api.sql_validator import validate_query, ValidationError
validate_query("SELECT * FROM predictions WHERE decision = 'FLAG'")  # ✅
validate_query("SELECT * FROM users; DROP TABLE predictions")         # ❌ raises
```

### 3. Human-in-the-Loop Checkpoints

Fraud analyst review queue — all FLAG decisions sorted by risk:

```bash
curl http://localhost:8001/api/v1/flagged
```

Returns: transaction details + flag reason + minutes waiting for review.

### 4. MCP Enterprise Integration

Mock implementation of the Model Context Protocol pattern for Slack/Drive/Salesforce:

```bash
curl -X POST http://localhost:8001/api/v1/mcp/slack/alert \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn-001", "message": "High risk detected", "channel": "#fraud-alerts"}'
```

---

## Monitoring & Observability

### Backend Ops Dashboard (Grafana)

Access: **http://localhost:3000** (admin/admin)

4-row dashboard auto-provisioned from `monitoring/backend-ops-dashboard.json`:

| Row | Panels |
|-----|--------|
| System Health | API Uptime, HTTP Response Codes, Error Rate, Containers Up |
| Performance | Latency p50/p95/p99, Kafka Lag, PostgreSQL Connections |
| Business Metrics | Transactions/sec, Decision Distribution (pie), Hourly Trend |
| Model Monitoring | Predict vs SHAP Latency, Feature Usage |

See [monitoring/README.md](monitoring/README.md) for full instructions.

### Prometheus Metrics

Scrape target: `http://localhost:8001/api/metrics/prometheus`

---

## AI/ML Enhancements

### Dual LLM Explanations (Qwen vs GPT-4o-mini)

```bash
# Default: Qwen 2.5 (free, local via Ollama)
curl -X POST http://localhost:8001/api/v1/explain \
  -d '{"transaction_id": "txn-001", "features": {...}}'

# OpenAI GPT-4o-mini
curl -X POST "http://localhost:8001/api/v1/explain?model=openai" \
  -d '{"transaction_id": "txn-001", "features": {...}}'

# Side-by-side comparison (both models)
curl -X POST "http://localhost:8001/api/v1/explain?compare=true" \
  -d '{"transaction_id": "txn-001", "features": {...}}'
```

The React dashboard shows a "Compare with GPT-4o-mini" button in the SHAP Explainer panel.
Click once to load both explanations side-by-side. Result is cached (button disabled after use).

### pgvector Similarity Search

Find past frauds that look like the current transaction:

```bash
curl http://localhost:8001/api/v1/similar/txn-001
```

Uses HNSW cosine similarity on 498-dimensional feature embeddings stored in PostgreSQL.
Requires pgvector extension (included in `docker/Dockerfile.postgres`).

### Data Augmentation for Robustness

```bash
# Mix 80% real + 20% synthetic frauds in the Kafka stream
python -m src.streaming.kafka_producer --augment --rate 200 --count 1000
```

Generates 5 fraud patterns: `high_amount`, `device_sharing`, `rapid_sequence`,
`cross_border`, `round_number`. Synthetic messages are tagged with `metadata.synthetic=true`.

---

## Production Deployment

### GitOps with ArgoCD

```bash
# Register with ArgoCD
kubectl apply -f deployment/argocd-app.yaml

# ArgoCD watches main branch and auto-deploys on every merge
argocd app sync atlas-x
```

See [deployment/README.md](deployment/README.md) for full GitOps workflow,
rollback procedure, and secrets management.

### Kubernetes Manifests

```
deployment/k8s/
├── api-deployment.yaml      # FastAPI + HPA (2–10 replicas)
├── kafka-deployment.yaml
├── neo4j-statefulset.yaml
├── postgres-statefulset.yaml  # pgvector/pgvector:pg16
├── ingress.yaml
├── configmap.yaml
└── secrets.yaml             # TEMPLATE – use Sealed Secrets in production
```

---

## API Documentation

See [docs/API.md](docs/API.md) for full endpoint reference with curl examples.

Interactive Swagger UI: `http://localhost:8001/docs`

---

## License

MIT — see `LICENSE`.

---

## Author

Thayaananthan Kanagaraj. 
Xiu-Wen Yeh (Vian). 
Muskaan Nolastname. 
Karthikeyan Sivasubramanian. 