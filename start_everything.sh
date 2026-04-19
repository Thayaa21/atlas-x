#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting ATLAS-X Production System..."

# ── Load .env ─────────────────────────────────────────────────────────────────
if [ -f ".env" ]; then
  while IFS='=' read -r key val; do
    key=$(echo "$key" | tr -d ' ')
    val=$(echo "$val" | tr -d ' ')
    [[ -z "$key" || "$key" == \#* ]] && continue
    export "$key=$val"
  done < .env
  echo "[env] Loaded .env"
fi

# ── Kill existing ─────────────────────────────────────────────────────────────
echo "[0/5] Cleaning up existing processes..."
pkill -f uvicorn 2>/dev/null || true
pkill -f kafka_consumer 2>/dev/null || true
pkill -f kafka_producer 2>/dev/null || true
pkill -f vite 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:5180 | xargs kill -9 2>/dev/null || true
sleep 2

# ── Docker (Kafka, Zookeeper, Neo4j, Redis, Postgres) ────────────────────────
echo "[1/5] Starting Docker services..."
cd docker && docker-compose up -d && cd ..
echo "      Waiting 20s for services to be ready..."
sleep 20

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "[venv] Activated .venv"
fi

# ── API ───────────────────────────────────────────────────────────────────────
echo "[2/5] Starting API on :8001..."
./run_api.sh > /tmp/atlas_api.log 2>&1 &
API_PID=$!
echo "      Waiting for API to be healthy..."
until curl -s http://localhost:8001/api/v1/health | grep -q "healthy" 2>/dev/null; do
  sleep 2
done
echo "      API is up ✓"

# ── Kafka consumer ────────────────────────────────────────────────────────────
echo "[3/5] Starting Kafka consumer..."
python -m src.streaming.kafka_consumer > /tmp/atlas_consumer.log 2>&1 &
CONSUMER_PID=$!
sleep 3

# ── Kafka producer (infinite loop) ───────────────────────────────────────────
echo "[4/5] Starting Kafka producer (infinite loop, 200 txns/batch)..."
python -m src.streaming.kafka_producer --loop --rate 50 --count 200 > /tmp/atlas_producer.log 2>&1 &
PRODUCER_PID=$!

# ── Cleanup on Ctrl-C ─────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "🛑 Shutting down ATLAS-X..."
  kill $API_PID $CONSUMER_PID $PRODUCER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Frontend (foreground — keeps script alive) ────────────────────────────────
echo "[5/5] Starting dashboard on http://localhost:5180 ..."
echo ""
echo "  Dashboard  → http://localhost:5180"
echo "  API docs   → http://localhost:8001/docs"
echo "  Logs       → /tmp/atlas_api.log  /tmp/atlas_producer.log"
echo ""
cd frontend && npm run dev
