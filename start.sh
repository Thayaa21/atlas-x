#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ATLAS-X  start.sh  —  Start everything cleanly
#
# Order of operations (each step waits for the previous to be healthy):
#   1. Docker services  (Kafka, Neo4j, Redis, Postgres, Prometheus, Grafana)
#   2. FastAPI          (waits for /api/v1/health)
#   3. Kafka consumer   (waits for API to be healthy first)
#   4. Kafka producer   (starts last — feeds data into the consumer)
#   5. React frontend
#
# PIDs are written to .atlas_pids so stop.sh can kill them cleanly.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$REPO/.atlas_pids"
LOG_DIR="$REPO/.atlas_logs"

# ── Colours ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; R='\033[0;31m'; N='\033[0m'
ok()   { echo -e "${G}[✓]${N} $*"; }
info() { echo -e "${C}[→]${N} $*"; }
warn() { echo -e "${Y}[!]${N} $*"; }
die()  { echo -e "${R}[✗]${N} $*"; exit 1; }

# ── Guard: don't double-start ─────────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    warn "ATLAS-X may already be running (.atlas_pids exists)."
    warn "Run ./stop.sh first, or delete .atlas_pids if it's stale."
    exit 1
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"   # create empty after guard passes

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ATLAS-X  —  Starting Up                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Load .env (handles KEY = value with spaces) ────────────────────────────
if [ -f "$REPO/.env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        key="${line%%=*}"; val="${line#*=}"
        key="${key// /}";  val="${val# }"; val="${val% }"
        export "$key=$val"
    done < "$REPO/.env"
    ok "Loaded .env"
else
    warn ".env not found — OPENAI_API_KEY and POSTGRES_DSN may be unset"
fi

# ── 2. Activate venv ──────────────────────────────────────────────────────────
if [ -f "$REPO/.venv/bin/activate" ]; then
    source "$REPO/.venv/bin/activate"
    ok "Activated .venv"
elif [ -f "$REPO/venv/bin/activate" ]; then
    source "$REPO/venv/bin/activate"
    ok "Activated venv"
else
    warn "No virtual environment found — using system Python"
fi

# ── 3. Docker services ────────────────────────────────────────────────────────
info "Starting Docker services…"
# Start existing containers if they exist, otherwise create them
# Note: docker_default network is recreated by compose if missing
RUNNING=$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')
if [ "$RUNNING" -gt 0 ]; then
    info "Docker containers already running ($RUNNING containers)"
else
    # Check if containers exist but are stopped
    EXISTING=$(docker ps -aq --filter "name=docker-" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$EXISTING" -gt 0 ]; then
        # Try to start existing containers; if network is missing, recreate via compose
        docker start $(docker ps -aq --filter "name=docker-") 2>/dev/null || {
            warn "Network missing — recreating containers via compose…"
            docker container prune -f 2>/dev/null
            docker compose -f "$REPO/docker/docker-compose.yml" up -d --no-deps \
                zookeeper kafka neo4j redis postgres prometheus grafana 2>&1 \
                | grep -E "Started|Created|Error" || true
        }
    else
        docker compose -f "$REPO/docker/docker-compose.yml" up -d --no-deps \
            zookeeper kafka neo4j redis postgres prometheus grafana 2>&1 \
            | grep -E "Started|Created|Error" || true
    fi
fi

# Wait for Kafka to be ready (it takes the longest)
info "Waiting for Kafka to be ready…"
for i in $(seq 1 30); do
    if docker exec docker-kafka-1 \
        kafka-topics --bootstrap-server localhost:9092 --list &>/dev/null 2>&1; then
        ok "Kafka ready"
        break
    fi
    [ "$i" -eq 30 ] && die "Kafka did not start in 60s. Check: docker compose -f docker/docker-compose.yml logs kafka"
    sleep 2
done

# Wait for Postgres (Docker, port 5433)
info "Waiting for PostgreSQL (port 5433)…"
for i in $(seq 1 20); do
    if docker exec docker-postgres-1 pg_isready -U postgres &>/dev/null 2>&1; then
        ok "PostgreSQL ready"
        break
    fi
    [ "$i" -eq 20 ] && warn "PostgreSQL not ready — continuing anyway"
    sleep 2
done

ok "Docker services up"

# ── 4. FastAPI ────────────────────────────────────────────────────────────────
# Kill anything already on port 8001
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
sleep 1

info "Starting FastAPI on port 8001…"
nohup python3 -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --log-level warning \
    > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
echo "API=$API_PID" >> "$PID_FILE"

# Wait for API health (up to 60s — model loading takes ~5s)
info "Waiting for API to be healthy…"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8001/api/v1/health &>/dev/null; then
        ok "FastAPI ready  →  http://localhost:8001"
        ok "Swagger docs   →  http://localhost:8001/docs"
        break
    fi
    [ "$i" -eq 30 ] && die "API did not start in 60s. Check: tail -50 $LOG_DIR/api.log"
    sleep 2
done

# ── 5. Kafka consumer ─────────────────────────────────────────────────────────
# Start AFTER API is healthy — consumer calls /api/v1/predict immediately
info "Starting Kafka consumer…"
nohup python3 -m src.streaming.kafka_consumer \
    > "$LOG_DIR/consumer.log" 2>&1 &
CONSUMER_PID=$!
echo "CONSUMER=$CONSUMER_PID" >> "$PID_FILE"

# Give consumer 5s to connect
sleep 5
if kill -0 "$CONSUMER_PID" 2>/dev/null; then
    ok "Kafka consumer running (PID $CONSUMER_PID)"
else
    warn "Kafka consumer exited early — check $LOG_DIR/consumer.log"
fi

# ── 6. Kafka producer ─────────────────────────────────────────────────────────
info "Starting Kafka producer (20 txns/sec, loop)…"
nohup python3 -m src.streaming.kafka_producer \
    --rate 20 --count 0 --loop \
    > "$LOG_DIR/producer.log" 2>&1 &
PRODUCER_PID=$!
echo "PRODUCER=$PRODUCER_PID" >> "$PID_FILE"

sleep 3
if kill -0 "$PRODUCER_PID" 2>/dev/null; then
    ok "Kafka producer running (PID $PRODUCER_PID)"
else
    warn "Kafka producer exited early — check $LOG_DIR/producer.log"
fi

# ── 7. React frontend ─────────────────────────────────────────────────────────
if command -v node &>/dev/null; then
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    sleep 1
    info "Starting React dashboard on port 5173…"
    cd "$REPO/frontend"
    nohup npm run dev -- --port 5173 \
        > "$LOG_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo "FRONTEND=$FRONTEND_PID" >> "$PID_FILE"
    cd "$REPO"
    sleep 4
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        ok "React dashboard running (PID $FRONTEND_PID)  →  http://localhost:5173"
    else
        warn "Frontend exited early — check $LOG_DIR/frontend.log"
    fi
else
    warn "Node.js not found — skipping frontend"
fi

# ── 8. Ollama check ───────────────────────────────────────────────────────────
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    ok "Ollama running  →  Qwen 2.5 LLM explanations available"
else
    warn "Ollama not running — AI explanations will fall back to OpenAI"
    warn "To start: ollama serve  (in a separate terminal)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  ATLAS-X is Running                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  User Dashboard    →  http://localhost:5173              ║"
echo "║  API               →  http://localhost:8001              ║"
echo "║  Swagger Docs      →  http://localhost:8001/docs         ║"
echo "║  Grafana (Ops)     →  http://localhost:3000  admin/admin ║"
echo "║  Prometheus        →  http://localhost:9090              ║"
echo "║  Neo4j Browser     →  http://localhost:7474              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Logs:  .atlas_logs/api.log                              ║"
echo "║         .atlas_logs/consumer.log                         ║"
echo "║         .atlas_logs/producer.log                         ║"
echo "║         .atlas_logs/frontend.log                         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  To stop:  ./stop.sh                                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
