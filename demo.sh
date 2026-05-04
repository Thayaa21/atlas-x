#!/bin/bash
# ATLAS-X Demo Script
# ─────────────────────────────────────────────────────────────────────────────
# Demonstrates the full API: health check, stats, fraud rings, and scores
# the 6 sample transactions included in tests/sample_transactions.json.
#
# Prerequisites:
#   ./run_api.sh  (API must be running on localhost:8000)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

API="http://localhost:8001"
SAMPLES="tests/sample_transactions.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

sep() { echo -e "${BLUE}────────────────────────────────────────────────────────${NC}"; }

echo ""
echo -e "${BOLD}  ATLAS-X Fraud Detection — Live Demo${NC}"
sep

# ── Check API is up ────────────────────────────────────────────────────────────

echo -e "\n${CYAN}[1/5] Health check${NC}"
HEALTH=$(curl -sf "$API/api/v1/health" 2>/dev/null) || {
    echo -e "${RED}ERROR: API not reachable at $API${NC}"
    echo "       Start it with:  ./run_api.sh"
    exit 1
}
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"

# ── Runtime stats ──────────────────────────────────────────────────────────────

echo -e "\n${CYAN}[2/5] Runtime statistics${NC}"
STATS=$(curl -sf "$API/api/v1/stats" 2>/dev/null)
echo "$STATS" | python3 -m json.tool 2>/dev/null || echo "$STATS"

# ── Fraud rings ────────────────────────────────────────────────────────────────

echo -e "\n${CYAN}[3/5] Fraud rings (min_cards=3, min_frauds=2)${NC}"
RINGS=$(curl -sf "$API/api/v1/graph/rings?min_cards=3&min_frauds=2" 2>/dev/null)
RING_TOTAL=$(echo "$RINGS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total',0))" 2>/dev/null || echo "?")
if [ "$RING_TOTAL" = "0" ] || [ "$RING_TOTAL" = "?" ]; then
    echo -e "${YELLOW}  No rings found (Neo4j may be unavailable or data not loaded)${NC}"
else
    echo -e "  ${GREEN}Found $RING_TOTAL fraud ring(s)${NC}"
    echo "$RINGS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for r in d.get('rings', [])[:5]:
    print(f\"  {r['ring_type']:8s}  ring={r['ring_id'][:16]:16s}  cards={r['card_count']:3d}  fraud={r['fraud_count']:3d}  rate={r['fraud_rate']*100:.1f}%\")
" 2>/dev/null || echo "$RINGS"
fi

# ── Score sample transactions ─────────────────────────────────────────────────

echo -e "\n${CYAN}[4/5] Scoring sample transactions${NC}"

if [ ! -f "$SAMPLES" ]; then
    echo -e "${YELLOW}  $SAMPLES not found — skipping transaction scoring${NC}"
else
    SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$SAMPLES'))))" 2>/dev/null || echo "?")
    echo "  Sending $SAMPLE_COUNT transactions from $SAMPLES..."
    echo ""

    python3 - <<'PYEOF'
import json, urllib.request, urllib.error, sys

SAMPLES_PATH = "tests/sample_transactions.json"
API_BASE     = "http://localhost:8001"

with open(SAMPLES_PATH) as f:
    samples = json.load(f)

DECISION_COLOR = {"BLOCK": "\033[0;31m", "FLAG": "\033[1;33m", "APPROVE": "\033[0;32m"}
RESET = "\033[0m"

for i, txn in enumerate(samples):
    payload = json.dumps(txn).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        decision = result.get("decision", "?")
        prob     = result.get("fraud_probability", 0.0)
        graph    = result.get("graph_risk_score", 0.0)
        seg      = result.get("customer_segment", "?")
        lat      = result.get("latency_ms", 0.0)
        txn_id   = result.get("transaction_id", "?")
        color    = DECISION_COLOR.get(decision, "")
        print(f"  Txn {i+1:2d}  id={txn_id:12s}  {color}{decision:7s}{RESET}  "
              f"p={prob:.4f}  graph={graph:.4f}  seg={seg:12s}  {lat:.1f}ms")
    except Exception as e:
        print(f"  Txn {i+1:2d}  ERROR: {e}")
PYEOF
fi

# ── SHAP explanation for first transaction ────────────────────────────────────

echo -e "\n${CYAN}[5/5] SHAP explanation (first sample transaction)${NC}"

if [ -f "$SAMPLES" ]; then
    python3 - <<'PYEOF'
import json, urllib.request, sys

with open("tests/sample_transactions.json") as f:
    samples = json.load(f)

first = samples[0]
txn_id = str(first.get("TransactionID", "demo"))

payload = json.dumps({
    "transaction_id": txn_id,
    "features": first,
}).encode()

req = urllib.request.Request(
    "http://localhost:8001/api/v1/explain",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    print(f"  Transaction: {result['transaction_id']}")
    print(f"  Reasoning:   {result['decision_reasoning']}")
    print(f"  Graph:       {result['graph_explanation']}")
    print(f"\n  Top SHAP features (positive = pushes toward FRAUD):")
    shap = result.get("shap_values", {})
    for feat, val in list(shap.items())[:10]:
        bar = "█" * int(abs(val) * 60)
        sign = "+" if val >= 0 else "-"
        color = "\033[0;31m" if val >= 0 else "\033[0;32m"
        reset = "\033[0m"
        print(f"  {feat:30s}  {color}{sign}{abs(val):.5f}  {bar[:25]}{reset}")
except Exception as e:
    print(f"  SHAP unavailable: {e}")
    print("  (SHAP requires the full feature dict — model must be loaded)")
PYEOF
else
    echo -e "${YELLOW}  Skipped (no sample transactions file)${NC}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

sep
echo ""
echo -e "${GREEN}${BOLD}  Demo complete.${NC}"
echo ""
echo "  Dashboard:      http://localhost:5173"
echo "  API docs:       http://localhost:8001/docs"
echo "  Prometheus:     http://localhost:9090"
echo ""
echo "  Stream live transactions:"
echo "    python -m src.streaming.kafka_producer --rate 200 --count 500"
echo "    python -m src.streaming.kafka_consumer"
echo ""
