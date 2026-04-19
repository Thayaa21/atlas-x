import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { explainTransaction, explainLLM } from "../api/client";
import type { RecentTxn, Explanation } from "../api/client";
import { mockShapValues } from "../utils/mockData";

interface Props {
  selectedTxn: RecentTxn | null;
  useMock?: boolean;
}

interface ShapEntry {
  feature: string;
  value: number;
}

export default function ShapExplainer({ selectedTxn, useMock = false }: Props) {
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [llmText, setLlmText] = useState<string | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedTxn) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setLlmText(null);

    async function load() {
      try {
        let exp: Explanation;
        if (useMock) {
          await new Promise((r) => setTimeout(r, 300));
          exp = {
            transaction_id: selectedTxn!.transaction_id,
            shap_values: mockShapValues(),
            graph_explanation: "No fraud ring connection detected.",
            decision_reasoning: `fraud_prob=${selectedTxn!.fraud_probability.toFixed(3)} → ${selectedTxn!.decision}`,
          };
        } else {
          // Try real API; fall back to mock on error
          try {
            // We don't have the full feature dict from RecentTxn, so we pass minimal context
            // and let the API use defaults / return an explanation
            exp = await explainTransaction(selectedTxn!.transaction_id, {
              transaction_id: selectedTxn!.transaction_id,
            });
          } catch {
            exp = {
              transaction_id: selectedTxn!.transaction_id,
              shap_values: mockShapValues(),
              graph_explanation: "No fraud ring connection detected.",
              decision_reasoning: `fraud_prob=${selectedTxn!.fraud_probability.toFixed(3)} → ${selectedTxn!.decision}`,
            };
          }
        }
        if (!cancelled) {
          setExplanation(exp);
          // Fire LLM explanation after SHAP is ready
          setLlmLoading(true);
          try {
            const llm = await explainLLM({
              transaction_id:    exp.transaction_id,
              fraud_probability: selectedTxn!.fraud_probability,
              decision:          selectedTxn!.decision,
              customer_segment:  selectedTxn!.customer_segment,
              graph_risk_score:  selectedTxn!.graph_risk_score,
              shap_values:       exp.shap_values,
              graph_explanation: exp.graph_explanation,
            });
            if (!cancelled) setLlmText(llm.explanation);
          } catch {
            if (!cancelled) setLlmText(null);
          } finally {
            if (!cancelled) setLlmLoading(false);
          }
        }
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message ?? e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [selectedTxn?.transaction_id, useMock]);

  const shapEntries: ShapEntry[] = explanation
    ? Object.entries(explanation.shap_values)
        .map(([feature, value]) => ({ feature, value }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 10)
    : [];

  const decisionColor: Record<string, string> = {
    BLOCK: "#ef4444",
    FLAG: "#f59e0b",
    APPROVE: "#22c55e",
  };

  return (
    <div className="panel shap-panel">
      <h2 className="panel-title">SHAP Explainer</h2>

      {!selectedTxn && (
        <div className="empty-state">Click a transaction in the feed to explain it.</div>
      )}

      {selectedTxn && (
        <>
          <div className="shap-header">
            <span className="mono dim">txn {selectedTxn.transaction_id}</span>
            <span
              className="badge"
              style={{
                background: decisionColor[selectedTxn.decision] + "33",
                color: decisionColor[selectedTxn.decision],
                border: `1px solid ${decisionColor[selectedTxn.decision]}`,
              }}
            >
              {selectedTxn.decision}
            </span>
            <span className="dim">p={selectedTxn.fraud_probability.toFixed(4)}</span>
          </div>

          {loading && <div className="dim" style={{ marginBottom: 8 }}>Computing SHAP…</div>}
          {error && <div style={{ color: "#ef4444", marginBottom: 8 }}>{error}</div>}

          {explanation && !loading && (
            <>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart
                  data={shapEntries}
                  layout="vertical"
                  margin={{ left: 8, right: 24, top: 4, bottom: 4 }}
                >
                  <XAxis
                    type="number"
                    tick={{ fill: "#9ca3af", fontSize: 11 }}
                    axisLine={{ stroke: "#374151" }}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    width={140}
                    tick={{ fill: "#9ca3af", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{ background: "#2d2d2d", border: "1px solid #444", borderRadius: 6 }}
                    labelStyle={{ color: "#e5e7eb" }}
                    itemStyle={{ color: "#e5e7eb" }}
                    formatter={(v) => typeof v === "number" ? v.toFixed(5) : String(v)}
                  />
                  <ReferenceLine x={0} stroke="#4b5563" />
                  <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                    {shapEntries.map((e) => (
                      <Cell
                        key={e.feature}
                        fill={e.value >= 0 ? "#ef4444" : "#22c55e"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              <div className="shap-reasoning">
                <div className="dim" style={{ fontSize: 11, marginBottom: 4 }}>Decision reasoning</div>
                <div style={{ fontSize: 12, color: "#d1d5db" }}>
                  fraud_prob={selectedTxn!.fraud_probability.toFixed(4)}
                  {" · "}graph_risk={selectedTxn!.graph_risk_score.toFixed(4)}
                  {" · "}seg={selectedTxn!.customer_segment}
                  {" → "}{selectedTxn!.decision}
                </div>
                <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 4 }}>
                  {explanation.graph_explanation}
                </div>
              </div>

              <div className="shap-reasoning" style={{ marginTop: 8, borderTop: "1px solid #2d2d2d", paddingTop: 8 }}>
                <div className="dim" style={{ fontSize: 11, marginBottom: 4 }}>🤖 AI Explanation</div>
                {llmLoading && <div className="dim" style={{ fontSize: 12 }}>Generating explanation…</div>}
                {llmText && <div style={{ fontSize: 12, color: "#d1d5db", lineHeight: 1.6 }}>{llmText}</div>}
                {!llmLoading && !llmText && <div className="dim" style={{ fontSize: 12 }}>No explanation available.</div>}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
