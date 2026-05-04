# ATLAS-X Monitoring & Observability

## Backend Ops Dashboard (Grafana)

Access: **http://localhost:3000**  
Default credentials: `admin` / `admin`

The backend ops dashboard (`backend-ops-dashboard.json`) is auto-provisioned
by Grafana on startup via the volume mount in `docker/docker-compose.yml`.

### Dashboard Rows

| Row | Panels |
|-----|--------|
| **ROW 1 – System Health** | API Uptime, HTTP Response Codes, 5xx Error Rate, Containers Up |
| **ROW 2 – Performance Metrics** | API Latency p50/p95/p99, Kafka Consumer Lag, PostgreSQL Connections |
| **ROW 3 – Business Metrics** | Transactions Per Second, Decision Distribution (pie), Hourly Fraud Trend |
| **ROW 4 – Model Monitoring** | Predict vs SHAP Latency, Feature Usage (SHAP/Graph/pgvector) |

### Prometheus Datasource

Configured automatically via `docker/grafana/provisioning/datasources/datasource.yml`:

```
URL: http://prometheus:9090
```

Prometheus scrapes the FastAPI app at `http://api:8000/api/metrics/prometheus`
every 15 seconds (see `docker/prometheus/prometheus.yml`).

### Loading the Dashboard Manually

If the dashboard is not auto-loaded:

1. Open Grafana at http://localhost:3000
2. Go to **Dashboards → Import**
3. Upload `monitoring/backend-ops-dashboard.json`
4. Select **Prometheus** as the datasource
5. Click **Import**

### Key Prometheus Metrics (from FastAPI)

| Metric | Description |
|--------|-------------|
| `http_server_request_duration_seconds` | Request latency histogram (p50/p95/p99) |
| `http_server_request_duration_seconds_count` | Request count by route + status |
| `up{job="api"}` | API health (1=up, 0=down) |

### Adding Custom Metrics

To add business-level metrics (fraud rate, decision counts), instrument the
FastAPI app with `prometheus_client`:

```python
from prometheus_client import Counter, Histogram

fraud_decisions = Counter(
    "atlasx_decisions_total",
    "Fraud decisions by type",
    ["decision"]
)
# In _predict_core:
fraud_decisions.labels(decision=decision).inc()
```

## Prometheus

Access: **http://localhost:9090**

Config: `docker/prometheus/prometheus.yml`

## Alert Rules (Recommended)

Add to `docker/prometheus/prometheus.yml`:

```yaml
rule_files:
  - /etc/prometheus/alerts.yml
```

Example alerts:
- API error rate > 5% for 2 minutes
- p99 latency > 100ms for 5 minutes
- Kafka consumer lag > 1000 messages
- Fraud rate spike > 3× baseline
