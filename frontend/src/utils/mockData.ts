/**
 * Mock data generator — used when the API is offline (dev / demo mode).
 */

import type { Stats, RecentTxn, FraudRing } from "../api/client";

// ── helpers ────────────────────────────────────────────────────────────────

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min;
}
function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}
let _txCounter = 1_000_000;

// ── stats ──────────────────────────────────────────────────────────────────

export function mockStats(): Stats {
  const total = Math.floor(rand(800, 1200));
  const fraudCount = Math.floor(total * rand(0.04, 0.12));
  return {
    total_predictions: total,
    fraud_rate: parseFloat((fraudCount / total).toFixed(4)),
    avg_latency_ms: parseFloat(rand(12, 35).toFixed(2)),
    decisions: {
      APPROVE: total - fraudCount - Math.floor(fraudCount * 0.3),
      FLAG: Math.floor(fraudCount * 0.3),
      BLOCK: fraudCount,
    },
  };
}

// ── recent transactions ────────────────────────────────────────────────────

const SEGMENTS = ["premium", "standard", "new_account"];

export function mockRecentTxn(): RecentTxn {
  const decision = pick(["APPROVE", "APPROVE", "APPROVE", "FLAG", "BLOCK"]);
  const fraudProb =
    decision === "BLOCK"
      ? rand(0.7, 0.99)
      : decision === "FLAG"
      ? rand(0.3, 0.65)
      : rand(0.01, 0.18);
  const graphRisk =
    decision === "BLOCK"
      ? rand(0.5, 0.95)
      : rand(0.0, 0.3);
  return {
    transaction_id: String(_txCounter++),
    fraud_probability: parseFloat(fraudProb.toFixed(4)),
    graph_risk_score: parseFloat(graphRisk.toFixed(4)),
    customer_segment: pick(SEGMENTS),
    decision,
    confidence: parseFloat(rand(0.6, 0.99).toFixed(4)),
    latency_ms: parseFloat(rand(8, 60).toFixed(2)),
    timestamp: new Date().toISOString(),
  };
}

export function mockRecentList(n = 20): RecentTxn[] {
  return Array.from({ length: n }, () => mockRecentTxn()).sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
}

// ── fraud rings ────────────────────────────────────────────────────────────

const RING_TYPES = ["device", "email", "address"];

export function mockFraudRings(n = 8): FraudRing[] {
  return Array.from({ length: n }, (_, i) => {
    const cardCount = Math.floor(rand(4, 25));
    const fraudCount = Math.floor(rand(2, cardCount));
    return {
      ring_type: pick(RING_TYPES),
      ring_id: `mock-ring-${i + 1}`,
      card_count: cardCount,
      fraud_count: fraudCount,
      fraud_rate: parseFloat((fraudCount / cardCount).toFixed(4)),
    };
  }).sort((a, b) => b.fraud_count - a.fraud_count);
}

// ── SHAP values ────────────────────────────────────────────────────────────

const TOP_FEATURES = [
  "TransactionAmt",
  "graph_risk_score",
  "device_fraud_rate",
  "D1",
  "D15",
  "card1",
  "addr1",
  "dist1",
  "email_fraud_rate",
  "connected_fraud_cards",
];

export function mockShapValues(): Record<string, number> {
  const result: Record<string, number> = {};
  for (const f of TOP_FEATURES) {
    result[f] = parseFloat((rand(-0.3, 0.5) * (Math.random() > 0.5 ? 1 : -1)).toFixed(5));
  }
  return result;
}
