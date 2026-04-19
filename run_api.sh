#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ATLAS-X API Startup ==="

# ── Neo4j ─────────────────────────────────────────────────────────────────────
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "neo4j-fraud"; then
  echo "[neo4j] Container not running."
  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "neo4j-fraud"; then
    echo "[neo4j] Starting existing container..."
    docker start neo4j-fraud
    echo "[neo4j] Waiting 10s for Neo4j to accept connections..."
    sleep 10
  else
    echo "[neo4j] Container 'neo4j-fraud' not found — skipping (graph features will be disabled)."
  fi
else
  echo "[neo4j] Already running."
fi

# ── Environment variables ─────────────────────────────────────────────────────
if [ -f ".env" ]; then
  while IFS='=' read -r key val; do
    key=$(echo "$key" | tr -d ' ')
    val=$(echo "$val" | tr -d ' ')
    [[ -z "$key" || "$key" == \#* ]] && continue
    export "$key=$val"
  done < .env
  echo "[env] Loaded .env"
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "[venv] Activated .venv"
fi

# ── Model files check ─────────────────────────────────────────────────────────
if [ ! -f "src/models/atlass_x_xgb_v4_graph.pkl" ]; then
  echo "[ERROR] v4 model not found: src/models/atlass_x_xgb_v4_graph.pkl"
  echo "        Run: python -m src.models.train_v4_with_graph"
  exit 1
fi

if [ ! -f "src/optimization/artifacts/thresholds.json" ]; then
  echo "[ERROR] Threshold artifact not found: src/optimization/artifacts/thresholds.json"
  exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
RELOAD_FLAG="--reload"
if [ "$ATLAS_ENV" = "production" ]; then
  RELOAD_FLAG=""
  WORKERS="--workers 2"
else
  WORKERS=""
fi

echo "[api] Starting FastAPI on http://0.0.0.0:8001 ..."
echo "[api] Docs: http://localhost:8001/docs"
echo "[api] Health: http://localhost:8001/api/v1/health"
echo ""

uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  $RELOAD_FLAG \
  $WORKERS