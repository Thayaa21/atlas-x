#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ATLAS-X  stop.sh  —  Stop everything cleanly
# ─────────────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$REPO/.atlas_pids"

G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; N='\033[0m'
ok()   { echo -e "${G}[✓]${N} $*"; }
info() { echo -e "${C}[→]${N} $*"; }
warn() { echo -e "${Y}[!]${N} $*"; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ATLAS-X  —  Shutting Down                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Kill Python processes from PID file ────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    info "Stopping processes from .atlas_pids…"
    while IFS='=' read -r name pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && ok "Stopped $name (PID $pid)"
        else
            warn "$name (PID $pid) was not running"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    ok "Removed .atlas_pids"
else
    warn ".atlas_pids not found — killing by port instead"
fi

# ── 2. Force-kill anything still on key ports ─────────────────────────────────
info "Cleaning up ports 8001 and 5173…"
lsof -ti:8001 | xargs kill -9 2>/dev/null && ok "Cleared port 8001" || true
lsof -ti:5173 | xargs kill -9 2>/dev/null && ok "Cleared port 5173" || true

# ── 3. Stop Docker services ───────────────────────────────────────────────────
info "Stopping Docker services…"
docker compose -f "$REPO/docker/docker-compose.yml" stop 2>&1 \
    | grep -E "Stopped|Error" || true
ok "Docker services stopped (data volumes preserved)"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ATLAS-X  —  Stopped                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Data is preserved in Docker volumes.                    ║"
echo "║  To start again:  ./start.sh                             ║"
echo "║  To wipe all data: docker compose -f docker/            ║"
echo "║    docker-compose.yml down -v                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
