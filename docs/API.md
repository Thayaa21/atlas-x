# ATLAS-X API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

All v1 endpoints are prefixed with `/api/v1/`.  
Requests and responses use `Content-Type: application/json`.

---

## Endpoints

- [POST /api/v1/predict](#post-apiv1predict)
- [POST /api/v1/explain](#post-apiv1explain)
- [GET /api/v1/graph/rings](#get-apiv1graphrings)
- [GET /api/v1/graph/card/{card_id}](#get-apiv1graphcardcard_id)
- [GET /api/v1/health](#get-apiv1health)
- [GET /api/v1/stats](#get-apiv1stats)
- [GET /api/v1/recent](#get-apiv1recent)

---

## POST /api/v1/predict

Score a transaction and return a fraud decision.

### Request body

```json
{
  "transaction_id": "string",
  "card_id":        "string",
  "amount":         123.45,
  "features": {
    "TransactionAmt": 123.45,
    "card1": 9500,
    "card2": 111.0,
    "card3": 150.0,
    "card5": 226.0,
    "addr1": 299.0,
    "addr2": 87.0,
    "dist1": 0.0,
    "dist2": -1.0,
    "D1": 2.0,
    "D15": 120.0,
    "ProductCD": "W",
    "DeviceType": "desktop",
    "P_emaildomain": "gmail.com",
    "... (498 total columns)": "..."
  }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `transaction_id` | string | yes | Unique identifier for the transaction |
| `card_id` | string | yes | Used for Neo4j graph lookup |
| `amount` | float > 0 | yes | Transaction amount in USD |
| `features` | object | yes | Full pre-computed feature dict (498 columns) |

### Response

```json
{
  "transaction_id":    "txn-001",
  "fraud_probability": 0.9791,
  "graph_risk_score":  0.6200,
  "customer_segment":  "New",
  "decision":          "BLOCK",
  "confidence":        0.9791,
  "latency_ms":        17.34,
  "timestamp":         "2026-04-15T10:23:44.123456+00:00"
}
```

| Field | Type | Notes |
|---|---|---|
| `fraud_probability` | float [0,1] | XGBoost P(fraud) |
| `graph_risk_score` | float [0,1] | Neo4j composite risk; 0 if Neo4j unavailable |
| `customer_segment` | string | `VIP` \| `Regular` \| `New` |
| `decision` | string | `APPROVE` \| `FLAG` \| `BLOCK` |
| `confidence` | float [0,1] | Distance from decision boundary |
| `latency_ms` | float | End-to-end inference time |

**Decision thresholds:**

| Decision | Condition |
|---|---|
| `BLOCK` | `fraud_probability >= segment_threshold` |
| `FLAG` | `fraud_probability >= threshold × 0.4` AND (in fraud ring OR `graph_risk >= 0.5`) |
| `APPROVE` | all others |

**Per-segment thresholds:** VIP=0.72, Regular=0.88, New=0.82

### curl example

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_transactions.json | python3 -m json.tool
```

---

## POST /api/v1/explain

Return SHAP top-10 feature attributions for a transaction.

### Request body

```json
{
  "transaction_id": "txn-001",
  "features": {
    "TransactionAmt": 123.45,
    "... (498 cols)": "..."
  }
}
```

The `features` dict is identical to the one sent to `/predict`.

### Response

```json
{
  "transaction_id":     "txn-001",
  "shap_values": {
    "TransactionAmt":       0.31240,
    "graph_risk_score":     0.28910,
    "device_fraud_rate":    0.21450,
    "D1":                  -0.09870,
    "D15":                 -0.07320,
    "card1":                0.05880,
    "addr1":               -0.04210,
    "dist1":                0.03100,
    "email_fraud_rate":     0.02940,
    "connected_fraud_cards": 0.02610
  },
  "graph_explanation":  "Card shares device with 3 confirmed fraud cards.",
  "decision_reasoning": "fraud_prob=0.979 vs threshold=0.820 (segment=New) → BLOCK; Card shares device with 3 confirmed fraud cards."
}
```

Positive SHAP values push toward fraud; negative values push toward legitimate.

### curl example

```bash
curl -s -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn-001", "features": {"TransactionAmt": 500, "D1": 0}}'
```

---

## GET /api/v1/graph/rings

List fraud rings detected in the Neo4j graph, ordered by fraud card count.

### Query parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_cards` | int (≥2) | 3 | Minimum cards in ring |
| `min_frauds` | int (≥1) | 2 | Minimum confirmed fraud cards |

### Response

```json
{
  "rings": [
    {
      "ring_type":   "device",
      "ring_id":     "dev-0099",
      "card_count":  18,
      "fraud_count": 11,
      "fraud_rate":  0.6111
    },
    {
      "ring_type":   "email",
      "ring_id":     "em-0047",
      "card_count":  7,
      "fraud_count": 5,
      "fraud_rate":  0.7143
    }
  ],
  "total": 2
}
```

Ring types: `device` (shared device ID), `email` (shared email domain),
`address` (shared billing address).

Returns HTTP 503 if Neo4j is unavailable.

### curl example

```bash
curl "http://localhost:8000/api/v1/graph/rings?min_cards=5&min_frauds=3"
```

---

## GET /api/v1/graph/card/{card_id}

Return the graph risk profile for a single card.

### Path parameter

| Parameter | Description |
|---|---|
| `card_id` | Card identifier string (e.g. `9500-111.0-150.0`) |

### Response

```json
{
  "card_id":          "9500-111.0-150.0",
  "connected_frauds": 4,
  "device_frauds":    3,
  "email_frauds":     1,
  "address_frauds":   0,
  "graph_risk_score": 0.6800,
  "in_fraud_ring":    true,
  "ring_type":        "device"
}
```

Returns HTTP 503 if Neo4j is unavailable.

### curl example

```bash
curl "http://localhost:8000/api/v1/graph/card/9500-111.0-150.0"
```

---

## GET /api/v1/health

Readiness probe — indicates whether the model and Neo4j are available.

### Response

```json
{
  "status":          "healthy",
  "xgboost_loaded":  true,
  "neo4j_connected": false,
  "uptime_seconds":  312.4
}
```

Returns HTTP 200 regardless of Neo4j status (model-only operation is valid).
Use `xgboost_loaded: false` to detect startup failures.

### curl example

```bash
curl http://localhost:8000/api/v1/health
```

---

## GET /api/v1/stats

Runtime prediction statistics since the last API restart.

### Response

```json
{
  "total_predictions": 1024,
  "fraud_rate":        0.0762,
  "avg_latency_ms":    17.31,
  "decisions": {
    "APPROVE": 946,
    "FLAG":    40,
    "BLOCK":   38
  }
}
```

### curl example

```bash
curl http://localhost:8000/api/v1/stats
```

---

## GET /api/v1/recent

Last 50 scored transactions (lightweight metadata only, no raw features).

### Response

```json
{
  "transactions": [
    {
      "transaction_id":    "txn-2041",
      "fraud_probability": 0.9791,
      "graph_risk_score":  0.6200,
      "customer_segment":  "New",
      "decision":          "BLOCK",
      "confidence":        0.9791,
      "latency_ms":        17.34,
      "timestamp":         "2026-04-15T10:23:44.123456+00:00"
    }
  ]
}
```

Ordered newest-first. Used by the React dashboard TransactionFeed panel.

### curl example

```bash
curl http://localhost:8000/api/v1/recent | python3 -m json.tool
```

---

## Legacy endpoints

These are kept for backward compatibility and map to the same logic as v1.

| Legacy | v1 equivalent |
|---|---|
| `POST /api/predict` | `POST /api/v1/predict` |
| `GET /api/graph/rings` | `GET /api/v1/graph/rings` |
| `GET /api/metrics` | `GET /api/v1/stats` |
| `GET /healthz` | `GET /api/v1/health` |
| `GET /api/metrics/prometheus` | Prometheus scrape target |
| `WS /ws/alerts` | Kafka → WebSocket fraud alert bridge |

---

## Error responses

| Status | Meaning |
|---|---|
| 422 | Validation error — missing or invalid request fields |
| 500 | Model inference error (check API logs) |
| 503 | Neo4j unavailable (graph endpoints only) |

Error body:

```json
{
  "detail": "Neo4j unavailable: Failed to connect to server..."
}
```
