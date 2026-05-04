import axios from "axios";

const http = axios.create({
  baseURL: "http://localhost:8001",
  timeout: 150_000,                           // 150s — Qwen on CPU can take ~20-30s
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
  // Feature 7: dual LLM fields (present when ?compare=true or ?model=...)
  qwen_explanation?: string | null;
  openai_explanation?: string | null;
  model_used?: string;
  generation_time_ms?: number;
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

// Feature 1: Event sourcing
export interface EventRecord {
  event_type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface EventHistory {
  transaction_id: string;
  events: EventRecord[];
  total: number;
}

// Feature 3: Flagged transactions
export interface FlaggedTransaction {
  transaction_id: string;
  fraud_probability: number;
  graph_risk_score: number;
  customer_segment: string;
  decision: string;
  latency_ms: number;
  created_at: string;
  minutes_waiting: number;
  flag_reason: string;
}

export interface FlaggedResponse {
  flagged: FlaggedTransaction[];
  total: number;
}

// Feature 6: Similarity search
export interface SimilarTransaction {
  transaction_id: string;
  similarity_score: number;
  is_fraud: boolean;
  created_at: string;
}

export interface SimilarityResponse {
  query_transaction_id: string;
  similar_transactions: SimilarTransaction[];
  total: number;
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
  features: Record<string, unknown>,
  options?: {
    model?: "qwen" | "openai";
    compare?: boolean;
    overrides?: {
      fraud_probability: number;
      decision: string;
      graph_risk_score: number;
      customer_segment: string;
    };
  }
): Promise<Explanation> {
  const params = new URLSearchParams();
  if (options?.model) params.set("model", options.model);
  if (options?.compare) params.set("compare", "true");
  const qs = params.toString() ? `?${params.toString()}` : "";

  const body: Record<string, unknown> = {
    transaction_id: transactionId,
    features,
  };

  // Pass real values as overrides so the LLM uses the correct fraud_probability
  // and decision instead of re-scoring from (possibly empty) features
  if (options?.overrides) {
    body.fraud_probability_override = options.overrides.fraud_probability;
    body.decision_override          = options.overrides.decision;
    body.graph_risk_override        = options.overrides.graph_risk_score;
    body.customer_segment_override  = options.overrides.customer_segment;
  }

  const { data } = await http.post<Explanation>(`/api/v1/explain${qs}`, body);
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

// Feature 1: Event sourcing
export async function getTransactionEvents(transactionId: string): Promise<EventHistory> {
  const { data } = await http.get<EventHistory>(`/api/v1/events/${encodeURIComponent(transactionId)}`);
  return data;
}

// Feature 3: Flagged transactions
export async function getFlaggedTransactions(limit = 50): Promise<FlaggedResponse> {
  const { data } = await http.get<FlaggedResponse>(`/api/v1/flagged?limit=${limit}`);
  return data;
}

// Feature 6: Similarity search
export async function getSimilarTransactions(
  transactionId: string,
  topK = 10
): Promise<SimilarityResponse> {
  const { data } = await http.get<SimilarityResponse>(
    `/api/v1/similar/${encodeURIComponent(transactionId)}?top_k=${topK}`
  );
  return data;
}

// Ground Truth Evaluation
export interface ConfusionMatrix { tp: number; fp: number; fn: number; tn: number; }
export interface EvalMetrics {
  precision: number; recall: number; f1: number;
  accuracy: number; fpr: number; auc: number | null;
}
export interface SegmentMetrics {
  n: number; n_fraud: number;
  precision: number; recall: number; f1: number;
  tp: number; fp: number; fn: number; tn: number;
  real_tp_amt?: number; real_fp_amt?: number; real_fn_amt?: number;
}
export interface ThresholdPoint {
  threshold: number; precision: number; recall: number; f1: number; flagged: number;
  real_tp_amt?: number; real_fp_amt?: number; real_fn_amt?: number;
  tp?: number; fp?: number; fn?: number;
}
export interface ScoreBucket {
  range: string; total: number; fraud: number; legit: number;
}
export interface EvaluationResponse {
  summary: {
    total_evaluated: number; actual_fraud: number; actual_legit: number;
    fraud_rate: number; avg_fraud_amt: number; avg_legit_amt: number;
  };
  active_threshold: number;
  threshold_source: "override" | "default";
  confusion_matrix: ConfusionMatrix;
  // Real dollar amounts from actual TransactionAmt in the dataset
  real_amounts: {
    tp_amt: number;   // actual fraud money caught (raw transaction value)
    fp_amt: number;   // legit money wrongly blocked
    fn_amt: number;   // actual fraud money missed (raw transaction value)
  };
  metrics: EvalMetrics;
  per_segment: Record<string, SegmentMetrics>;
  threshold_analysis: ThresholdPoint[];
  score_distribution: ScoreBucket[];
  flag_count?: number;
}

export async function getEvaluation(segment?: string): Promise<EvaluationResponse> {
  const qs = segment ? `?segment=${encodeURIComponent(segment)}` : "";
  const { data } = await http.get<EvaluationResponse>(`/api/v1/evaluation${qs}`);
  return data;
}

// Threshold optimization
export interface ThresholdCurvePoint {
  threshold: number; net_impact: number;
  tp_amt: number; fn_amt: number; fp_amt: number;
  tp: number; fp: number; fn: number;
  precision: number; recall: number; f1: number;
}

export interface OptimizeResponse {
  optimal_threshold: number;
  optimal_net_impact: number;
  current_threshold: number;
  current_net_impact: number;
  improvement: number;
  n_evaluated: number;
  curve: ThresholdCurvePoint[];
}

export async function optimizeThreshold(): Promise<OptimizeResponse> {
  const { data } = await http.get<OptimizeResponse>("/api/v1/threshold/optimize");
  return data;
}

export async function applyThreshold(threshold: number): Promise<{ status: string; threshold: number; message: string }> {
  const { data } = await http.post(`/api/v1/threshold/apply?threshold=${threshold}`);
  return data;
}

export async function resetThreshold(): Promise<{ status: string; thresholds: Record<string, number>; message: string }> {
  const { data } = await http.post("/api/v1/threshold/reset");
  return data;
}

export async function getCurrentThreshold(): Promise<{
  override_active: boolean; override_value: number | null;
  defaults: Record<string, number>; effective: Record<string, number>;
}> {
  const { data } = await http.get("/api/v1/threshold/current");
  return data;
}
