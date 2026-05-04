import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";
import { explainLLM, getSimilarTransactions } from "../api/client";
import type { RecentTxn, SimilarTransaction } from "../api/client";
import { mockShapValues } from "../utils/mockData";

interface Props {
  selectedTxn: RecentTxn | null;
  useMock?: boolean;
}

interface ShapEntry { feature: string; value: number; }

const DECISION_COLOR: Record<string, string> = {
  BLOCK: "#ef4444", FLAG: "#f59e0b", APPROVE: "#22c55e",
};

export default function ShapExplainer({ selectedTxn, useMock = false }: Props) {
  const [shapEntries, setShapEntries]         = useState<ShapEntry[]>([]);
  const [graphExplanation, setGraphExplanation] = useState("");
  const [decisionReasoning, setDecisionReasoning] = useState("");
  const [shapLoading, setShapLoading]         = useState(false);
  const [shapError, setShapError]             = useState<string | null>(null);

  // LLM state
  const [qwenText, setQwenText]               = useState("");
  const [openaiText, setOpenaiText]           = useState("");
  const [llmLoading, setLlmLoading]           = useState(false);
  const [showComparison, setShowComparison]   = useState(false);
  const [comparisonUsed, setComparisonUsed]   = useState(false);
  const [genMs, setGenMs]                     = useState<number | null>(null);

  // pgvector similar transactions state
  const [similar, setSimilar]                 = useState<SimilarTransaction[]>([]);
  const [similarLoading, setSimilarLoading]   = useState(false);
  const [showSimilar, setShowSimilar]         = useState(false);

  // ── Reset + load on new transaction ──────────────────────────────────────
  useEffect(() => {
    if (!selectedTxn) return;
    let cancelled = false;

    setShapEntries([]);
    setGraphExplanation("");
    setDecisionReasoning("");
    setShapError(null);
    setQwenText("");
    setOpenaiText("");
    setShowComparison(false);
    setComparisonUsed(false);
    setGenMs(null);
    setSimilar([]);
    setShowSimilar(false);

    // ── Step 1: Get SHAP values ─────────────────────────────────────────────
    async function loadShap() {
      setShapLoading(true);
      try {
        let shap: Record<string, number>;
        let graphExp = "No fraud ring connection detected.";
        let reasoning = `fraud_prob=${selectedTxn!.fraud_probability.toFixed(3)} → ${selectedTxn!.decision}`;

        if (useMock) {
          await new Promise(r => setTimeout(r, 300));
          shap = mockShapValues();
        } else {
          // Pass the real known values as overrides so the API uses them in
          // the LLM prompt, but still runs SHAP on the re-scored features.
          // We pass graph_risk_score as a feature so SHAP reflects it.
          const res = await fetch("http://localhost:8001/api/v1/explain?model=qwen", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              transaction_id: selectedTxn!.transaction_id,
              features: {
                graph_risk_score:  selectedTxn!.graph_risk_score,
                TransactionAmt:    selectedTxn!.confidence > 0.5 ? 100 : 500,
              },
              // Override values so LLM uses the real fraud_probability/decision
              fraud_probability_override: selectedTxn!.fraud_probability,
              decision_override:          selectedTxn!.decision,
              graph_risk_override:        selectedTxn!.graph_risk_score,
              customer_segment_override:  selectedTxn!.customer_segment,
            }),
          });

          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();

          shap     = data.shap_values ?? {};
          graphExp = data.graph_explanation ?? graphExp;
          reasoning = data.decision_reasoning ?? reasoning;

          // Use the Qwen explanation that came back with the SHAP call
          if (data.qwen_explanation && !cancelled) {
            setQwenText(data.qwen_explanation);
          }
        }

        if (!cancelled) {
          const entries = Object.entries(shap)
            .map(([feature, value]) => ({ feature, value }))
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
            .slice(0, 10);
          setShapEntries(entries);
          setGraphExplanation(graphExp);
          setDecisionReasoning(reasoning);
        }
      } catch (e: any) {
        if (!cancelled) {
          setShapError(String(e?.message ?? e));
          // Fallback to mock SHAP so chart isn't blank
          setShapEntries(
            Object.entries(mockShapValues())
              .map(([feature, value]) => ({ feature, value }))
              .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
              .slice(0, 10)
          );
        }
      } finally {
        if (!cancelled) setShapLoading(false);
      }
    }

    // ── Step 2: Get LLM explanation (uses real selectedTxn values) ──────────
    async function loadLlm() {
      if (useMock) return;
      setLlmLoading(true);
      try {
        const llm = await explainLLM({
          transaction_id:    selectedTxn!.transaction_id,
          fraud_probability: selectedTxn!.fraud_probability,   // ← real value
          decision:          selectedTxn!.decision,             // ← real value
          customer_segment:  selectedTxn!.customer_segment,
          graph_risk_score:  selectedTxn!.graph_risk_score,
          shap_values:       {},   // will be filled after SHAP loads
          graph_explanation: selectedTxn!.graph_risk_score > 0.4
            ? `Graph risk score is elevated at ${selectedTxn!.graph_risk_score.toFixed(3)}.`
            : "No significant fraud ring connection detected.",
        });
        if (!cancelled) setQwenText(llm.explanation);
      } catch {
        // SHAP call may have already set qwenText; don't overwrite with error
      } finally {
        if (!cancelled) setLlmLoading(false);
      }
    }

    loadShap();
    loadLlm();
    return () => { cancelled = true; };
  }, [selectedTxn?.transaction_id, useMock]);

  // ── Compare with GPT-4o-mini ──────────────────────────────────────────────
  async function handleCompare() {
    if (!selectedTxn || comparisonUsed) return;
    setLlmLoading(true);
    setComparisonUsed(true);
    const t0 = Date.now();
    try {
      const res = await fetch("http://localhost:8001/api/v1/explain?compare=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transaction_id:             selectedTxn.transaction_id,
          features:                   { graph_risk_score: selectedTxn.graph_risk_score },
          fraud_probability_override: selectedTxn.fraud_probability,
          decision_override:          selectedTxn.decision,
          graph_risk_override:        selectedTxn.graph_risk_score,
          customer_segment_override:  selectedTxn.customer_segment,
        }),
      });
      const data = await res.json();
      setQwenText(data.qwen_explanation ?? qwenText);
      setOpenaiText(data.openai_explanation ?? "");
      setGenMs(Date.now() - t0);
      setShowComparison(true);
    } catch (e: any) {
      setOpenaiText(`Comparison failed: ${e?.message ?? e}`);
      setShowComparison(true);
    } finally {
      setLlmLoading(false);
    }
  }

  // pgvector: load similar transactions on demand
  async function handleShowSimilar() {
    if (!selectedTxn || showSimilar) return;
    setSimilarLoading(true);
    setShowSimilar(true);
    try {
      const res = await getSimilarTransactions(selectedTxn.transaction_id, 8);
      setSimilar(res.similar_transactions);
    } catch {
      setSimilar([]);
    } finally {
      setSimilarLoading(false);
    }
  }

  if (!selectedTxn) {
    return (
      <div className="panel shap-panel">
        <h2 className="panel-title">SHAP Explainer</h2>
        <div className="empty-state">Click a transaction in the feed to explain it.</div>
      </div>
    );
  }

  const dc = DECISION_COLOR[selectedTxn.decision] ?? "#9ca3af";

  return (
    <div className="panel shap-panel">
      <h2 className="panel-title">SHAP Explainer</h2>

      {/* Header */}
      <div className="shap-header">
        <span className="mono dim">txn {selectedTxn.transaction_id}</span>
        <span className="badge" style={{
          background: dc + "33", color: dc, border: `1px solid ${dc}`,
        }}>
          {selectedTxn.decision}
        </span>
        <span className="dim">p={selectedTxn.fraud_probability.toFixed(4)}</span>
      </div>

      {/* SHAP chart */}
      {shapLoading && (
        <div className="dim" style={{ marginBottom: 8, fontSize: 12 }}>Computing SHAP…</div>
      )}
      {shapError && (
        <div style={{ color: "#f59e0b", fontSize: 11, marginBottom: 6 }}>
          ⚠ SHAP fallback (mock): {shapError}
        </div>
      )}

      {shapEntries.length > 0 && (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart
            data={shapEntries}
            layout="vertical"
            margin={{ left: 8, right: 24, top: 4, bottom: 4 }}
          >
            <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 11 }}
              axisLine={{ stroke: "#374151" }} />
            <YAxis type="category" dataKey="feature" width={150}
              tick={{ fill: "#9ca3af", fontSize: 11 }}
              axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#2d2d2d", border: "1px solid #444", borderRadius: 6 }}
              labelStyle={{ color: "#e5e7eb" }}
              itemStyle={{ color: "#e5e7eb" }}
              formatter={(v) => typeof v === "number" ? v.toFixed(5) : String(v)}
            />
            <ReferenceLine x={0} stroke="#4b5563" />
            <Bar dataKey="value" radius={[0, 3, 3, 0]}>
              {shapEntries.map((e) => (
                <Cell key={e.feature} fill={e.value >= 0 ? "#ef4444" : "#22c55e"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}

      {/* Decision reasoning */}
      <div className="shap-reasoning">
        <div className="dim" style={{ fontSize: 11, marginBottom: 4 }}>Decision reasoning</div>
        <div style={{ fontSize: 12, color: "#d1d5db" }}>
          fraud_prob={selectedTxn.fraud_probability.toFixed(4)}
          {" · "}graph_risk={selectedTxn.graph_risk_score.toFixed(4)}
          {" · "}seg={selectedTxn.customer_segment}
          {" → "}{selectedTxn.decision}
        </div>
        {graphExplanation && (
          <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 4 }}>
            {graphExplanation}
          </div>
        )}
      </div>

      {/* AI Explanation */}
      <div className="shap-reasoning" style={{
        marginTop: 8, borderTop: "1px solid #2d2d2d", paddingTop: 8,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
          <div className="dim" style={{ fontSize: 11 }}>
            🤖 AI Explanation
            {showComparison && genMs != null && (
              <span style={{ marginLeft: 8, color: "#6b7280" }}>({genMs}ms)</span>
            )}
          </div>
          <button
            onClick={handleCompare}
            disabled={comparisonUsed || llmLoading}
            style={{
              fontSize: 11, padding: "2px 8px", borderRadius: 4,
              border: "1px solid #4b5563",
              background: comparisonUsed ? "#1f2937" : "#374151",
              color: comparisonUsed ? "#6b7280" : "#e5e7eb",
              cursor: comparisonUsed ? "not-allowed" : "pointer",
            }}
          >
            {llmLoading ? "Loading…" : comparisonUsed ? "✓ Compared" : "Compare with GPT-4o-mini"}
          </button>
        </div>

        {llmLoading && !showComparison && (
          <div className="dim" style={{ fontSize: 12 }}>Generating explanation…</div>
        )}

        {showComparison ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <div style={{ background: "#1a1a2e", border: "1px solid #2d3748", borderRadius: 6, padding: "8px 10px" }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "#60a5fa", marginBottom: 4, textTransform: "uppercase" }}>
                🦙 Qwen 2.5 (Free Local)
              </div>
              <div style={{ fontSize: 11, color: "#d1d5db", lineHeight: 1.6 }}>
                {qwenText || <span className="dim">No explanation.</span>}
              </div>
            </div>
            <div style={{ background: "#0d1117", border: "1px solid #2d3748", borderRadius: 6, padding: "8px 10px" }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "#34d399", marginBottom: 4, textTransform: "uppercase" }}>
                ✨ GPT-4o-mini (OpenAI)
              </div>
              <div style={{ fontSize: 11, color: "#d1d5db", lineHeight: 1.6 }}>
                {openaiText || <span className="dim">No explanation.</span>}
              </div>
            </div>
          </div>
        ) : (
          <div>
            {!llmLoading && qwenText && (
              <div style={{ fontSize: 12, color: "#d1d5db", lineHeight: 1.6 }}>{qwenText}</div>
            )}
            {!llmLoading && !qwenText && (
              <div className="dim" style={{ fontSize: 12 }}>No explanation available.</div>
            )}
          </div>
        )}
      </div>

      {/* pgvector Similar Transactions */}
      {selectedTxn && (
        <div className="shap-reasoning" style={{
          marginTop: 8, borderTop: "1px solid #2d2d2d", paddingTop: 8,
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
            <div className="dim" style={{ fontSize: 11 }}>
              🔍 Similar Past Transactions (pgvector HNSW)
              <span style={{ marginLeft: 6, color: "#475569", fontSize: 10 }}>
                — nearest neighbours by feature similarity
              </span>
            </div>
            {!showSimilar && (
              <button onClick={handleShowSimilar} style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 4,
                border: "1px solid #4b5563", background: "#374151", color: "#e5e7eb",
                cursor: "pointer",
              }}>Find Similar</button>
            )}
          </div>
          {similarLoading && <div className="dim" style={{ fontSize: 12 }}>Searching embeddings…</div>}
          {showSimilar && !similarLoading && similar.length === 0 && (
            <div className="dim" style={{ fontSize: 12 }}>
              No embeddings yet — they populate as transactions are scored.
            </div>
          )}
          {similar.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {similar.map((s) => (
                <div key={s.transaction_id} style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "4px 8px", borderRadius: 4, background: "#1a1a2e",
                  border: `1px solid ${s.is_fraud ? "#ef444433" : "#22c55e33"}`,
                }}>
                  <span style={{ fontSize: 11, color: "#9ca3af", fontFamily: "monospace" }}>
                    {s.transaction_id}
                  </span>
                  <span style={{
                    fontSize: 10, padding: "1px 6px", borderRadius: 3,
                    background: s.is_fraud ? "#ef444422" : "#22c55e22",
                    color: s.is_fraud ? "#ef4444" : "#22c55e",
                  }}>
                    {s.is_fraud ? "FRAUD" : "LEGIT"}
                  </span>
                  <span style={{ fontSize: 11, color: "#6b7280" }}>
                    {(s.similarity_score * 100).toFixed(1)}% similar
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
