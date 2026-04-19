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

### 2. Start the API

```bash
./run_api.sh
```

FastAPI available at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

### 3. Start the frontend

```bash
cd frontend
npm run dev
```

Dashboard at `http://localhost:5173`.

### 4. Run a quick demo

```bash
./demo.sh
```

Sends sample transactions and prints scored results.

### 5. Start the streaming pipeline (optional)

```bash
# Consumer: scores transactions from Kafka → Postgres
python -m src.streaming.kafka_consumer

# Producer: replay holdout data at 200 txns/sec
python -m src.streaming.kafka_producer --rate 200 --count 1000

# Alert feed monitor
python -m src.streaming.monitor
```

---

## Project Structure

```
atlas-x/
├── data/
│   ├── raw/                        # IEEE-CIS CSV files (not committed)
│   └── processed/                  # Parquet feature files
├── docker/
│   ├── docker-compose.yml          # Kafka, Neo4j, Redis, Postgres, Grafana
│   └── prometheus/
├── frontend/
│   ├── src/
│   │   ├── api/client.ts           # axios API wrapper
│   │   ├── components/             # React dashboard panels
│   │   └── utils/mockData.ts       # offline mock data
│   ├── package.json
│   └── vite.config.ts
├── results/
│   ├── v4_graph_metrics.json
│   └── model_comparison.png
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app (6 v1 endpoints)
│   │   ├── model_loader.py         # singleton model + SHAP loader
│   │   ├── monitoring.py           # in-memory StatsTracker
│   │   └── schemas.py              # Pydantic request/response models
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
│       ├── alert_consumer.py
│       ├── init_db.py
│       └── monitor.py
├── tests/
│   ├── test_api_endpoints.py       # 8 API integration tests
│   ├── test_streaming.py           # end-to-end Kafka test
│   └── sample_transactions.json    # 6 holdout samples for tests
├── docs/
│   ├── API.md                      # endpoint reference
│   └── ARCHITECTURE.md             # detailed architecture
├── README.md
├── requirements.txt
├── setup.sh
├── run_api.sh
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
# API endpoint tests (requires API running on :8000)
python tests/test_api_endpoints.py

# Kafka streaming integration test (requires Kafka + Postgres)
python tests/test_streaming.py
```

Test suite: 8 API tests (predict fraud, predict legit, SHAP explain, graph rings,
graph card, stats, health, latency benchmark). All 8 pass with API latency p50 ≈ 15 ms.

---

## API Documentation

See [docs/API.md](docs/API.md) for full endpoint reference with curl examples.

Interactive Swagger UI: `http://localhost:8000/docs`

---

## License

MIT — see `LICENSE`.

---

## Author

Thayaananthan Kanagaraj  
[thayaa1903@gmail.com](mailto:thayaa1903@gmail.com)
