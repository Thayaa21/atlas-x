import axios from "axios";

const http = axios.create({
  baseURL: "http://localhost:8001",           // Vite proxies /api → localhost:8000
  timeout: 8_000,
  headers: { "Content-Type": "application/json" },
});

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Stats {
  total_predictions: number;
  fraud_rate: number;
  avg_latency_ms: number;
  decisions: Record<string, number>;
}

export interface RecentTxn {
  transaction_id: string;
  fraud_probability: number;
  graph_risk_score: number;
  customer_segment: string;
  decision: string;
  confidence: number;
  latency_ms: number;
  timestamp: string;
}

export interface FraudRing {
  ring_type: string;
  ring_id: string;
  card_count: number;
  fraud_count: number;
  fraud_rate: number;
}

export interface CardDetails {
  card_id: string;
  connected_frauds: number;
  device_frauds: number;
  email_frauds: number;
  address_frauds: number;
  graph_risk_score: number;
  in_fraud_ring: boolean;
  ring_type: string;
}

export interface Explanation {
  transaction_id: string;
  shap_values: Record<string, number>;
  graph_explanation: string;
  decision_reasoning: string;
}

export interface Prediction {
  transaction_id: string;
  fraud_probability: number;
  graph_risk_score: number;
  customer_segment: string;
  decision: string;
  confidence: number;
  latency_ms: number;
  timestamp: string;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function getStats(): Promise<Stats> {
  const { data } = await http.get<Stats>("/api/v1/stats");
  return data;
}

export async function getRecent(): Promise<RecentTxn[]> {
  const { data } = await http.get<{ transactions: RecentTxn[] }>("/api/v1/recent");
  return data.transactions ?? [];
}

export async function getFraudRings(
  minCards = 3,
  minFrauds = 2
): Promise<FraudRing[]> {
  const { data } = await http.get<{ rings: FraudRing[] }>(
    `/api/v1/graph/rings?min_cards=${minCards}&min_frauds=${minFrauds}`
  );
  return data.rings ?? [];
}

export async function getCardDetails(cardId: string): Promise<CardDetails> {
  const { data } = await http.get<CardDetails>(
    `/api/v1/graph/card/${encodeURIComponent(cardId)}`
  );
  return data;
}

export async function explainTransaction(
  transactionId: string,
  features: Record<string, unknown>
): Promise<Explanation> {
  const { data } = await http.post<Explanation>("/api/v1/explain", {
    transaction_id: transactionId,
    features,
  });
  return data;
}

export interface LLMExplanation {
  transaction_id: string;
  explanation: string;
}

export async function explainLLM(payload: {
  transaction_id: string;
  fraud_probability: number;
  decision: string;
  customer_segment: string;
  graph_risk_score: number;
  shap_values: Record<string, number>;
  graph_explanation: string;
}): Promise<LLMExplanation> {
  const { data } = await http.post<LLMExplanation>("/api/v1/explain/llm", payload);
  return data;
}

export async function predict(payload: {
  transaction_id: string;
  card_id: string;
  amount: number;
  features: Record<string, unknown>;
}): Promise<Prediction> {
  const { data } = await http.post<Prediction>("/api/v1/predict", payload);
  return data;
}
