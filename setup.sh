#!/bin/bash
# ATLAS-X Setup Script
# ─────────────────────────────────────────────────────────────────────────────
# Sets up the full ATLAS-X stack:
#   1. Python virtual environment + dependencies
#   2. Docker services (Kafka, Neo4j, Redis, Postgres)
#   3. Postgres schema
#   4. Frontend (Node.js) dependencies
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { echo -e "${BLUE}[ATLAS-X]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
die()  { echo -e "${RED}[ FAIL ]${NC} $*"; exit 1; }

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║     ATLAS-X  Setup                   ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── 1. Python virtual environment ─────────────────────────────────────────────

log "Step 1/4 — Python environment"

if [ ! -d "$REPO_ROOT/venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv "$REPO_ROOT/venv"
    ok "Virtual environment created at ./venv"
else
    ok "Virtual environment already exists"
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/venv/bin/activate"

log "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REPO_ROOT/requirements.txt"

# Ensure package is importable
touch "$REPO_ROOT/src/__init__.py"
touch "$REPO_ROOT/src/utils/__init__.py"
touch "$REPO_ROOT/src/data/__init__.py"
touch "$REPO_ROOT/src/models/__init__.py"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

ok "Python dependencies installed"

# ── 2. Docker services ────────────────────────────────────────────────────────

log "Step 2/4 — Docker services"

if ! command -v docker &>/dev/null; then
    die "Docker not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
fi

if ! docker info &>/dev/null; then
    die "Docker daemon is not running. Start Docker Desktop and try again."
fi

log "Starting infrastructure services (Kafka, Neo4j, Redis, Postgres)..."
cd "$REPO_ROOT/docker"
docker compose up -d zookeeper
sleep 5
docker compose up -d kafka neo4j redis postgres
cd "$REPO_ROOT"

ok "Waiting for services to be ready (15s)..."
sleep 15

# Check Kafka
if docker compose -f "$REPO_ROOT/docker/docker-compose.yml" ps kafka 2>/dev/null | grep -q "Up"; then
    ok "Kafka running  (localhost:29092)"
else
    warn "Kafka may not be ready yet — check: docker compose -f docker/docker-compose.yml ps"
fi

# Check Postgres
if docker compose -f "$REPO_ROOT/docker/docker-compose.yml" ps postgres 2>/dev/null | grep -q "Up"; then
    ok "Postgres running  (localhost:5432)"
else
    warn "Postgres may not be ready yet"
fi

# ── 3. Postgres schema ────────────────────────────────────────────────────────

log "Step 3/4 — Initialising Postgres schema"

if python -m src.streaming.init_db 2>/dev/null; then
    ok "Postgres tables created (predictions, fraud_alerts)"
else
    warn "Could not initialise Postgres schema (may already exist, or Postgres not ready)."
    warn "Run manually later:  python -m src.streaming.init_db"
fi

# ── 4. Frontend dependencies ──────────────────────────────────────────────────

log "Step 4/4 — Frontend dependencies"

if ! command -v node &>/dev/null; then
    warn "Node.js not found — skipping frontend setup."
    warn "Install Node.js 20+ from https://nodejs.org and run:  cd frontend && npm install"
else
    NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -lt 18 ]; then
        warn "Node.js $NODE_VER found — recommend Node.js 20+. Continuing..."
    fi
    log "Installing frontend packages..."
    cd "$REPO_ROOT/frontend"
    npm install --silent
    cd "$REPO_ROOT"
    ok "Frontend packages installed"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  Setup complete!  Next steps:                        ║"
echo "  ║                                                      ║"
echo "  ║  1. Start the API:                                   ║"
echo "  ║       ./run_api.sh                                   ║"
echo "  ║                                                      ║"
echo "  ║  2. Start the frontend:                              ║"
echo "  ║       cd frontend && npm run dev                     ║"
echo "  ║     → Dashboard:  http://localhost:5173              ║"
echo "  ║     → API docs:   http://localhost:8000/docs         ║"
echo "  ║                                                      ║"
echo "  ║  3. Run a quick demo:                                ║"
echo "  ║       ./demo.sh                                      ║"
echo "  ║                                                      ║"
echo "  ║  4. Stream transactions:                             ║"
echo "  ║       python -m src.streaming.kafka_producer         ║"
echo "  ║       python -m src.streaming.kafka_consumer         ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""
