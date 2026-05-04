import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell, Legend,
} from "recharts";
import { getEvaluation } from "../api/client";
import type { EvaluationResponse } from "../api/client";

const SEG_COLORS: Record<string, string> = {
  VIP: "#f59e0b", Regular: "#3b82f6", New: "#22c55e",
};

function MetricCard({
  label, value, sub, color = "#f9fafb", good, bad,
}: {
  label: string; value: string; sub?: string;
  color?: string; good?: boolean; bad?: boolean;
}) {
  const c = good ? "#4ade80" : bad ? "#f87171" : color;
  return (
    <div style={{
      background: "#0f172a", border: `1px solid ${good ? "#14532d" : bad ? "#7f1d1d" : "#1e293b"}`,
      borderRadius: 8, padding: "14px 16px", textAlign: "center",
    }}>
      <div style={{ fontSize: 10, color: "#64748b", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: c }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: "#64748b", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function ConfusionMatrix({ tp, fp, fn, tn }: { tp: number; fp: number; fn: number; tn: number }) {
  const total = tp + fp + fn + tn;
  const cell = (v: number, label: string, bg: string, textColor: string) => (
    <div style={{
      background: bg, borderRadius: 6, padding: "12px 8px",
      textAlign: "center", flex: 1,
    }}>
      <div style={{ fontSize: 20, fontWeight: 700, color: textColor }}>{v.toLocaleString()}</div>
      <div style={{ fontSize: 10, color: textColor + "aa", marginTop: 4 }}>{label}</div>
      <div style={{ fontSize: 10, color: textColor + "77" }}>{(v / total * 100).toFixed(1)}%</div>
    </div>
  );
  return (
    <div>
      <div style={{ fontSize: 10, color: "#64748b", marginBottom: 8, textAlign: "center" }}>
        Predicted BLOCK vs Actual isFraud
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        {cell(tp, "True Positive\n(Caught fraud)", "#14532d", "#4ade80")}
        {cell(fp, "False Positive\n(Wrong block)", "#7c2d12", "#fb923c")}
        {cell(fn, "False Negative\n(Missed fraud)", "#7f1d1d", "#f87171")}
        {cell(tn, "True Negative\n(Correct approve)", "#0f2d1a", "#86efac")}
      </div>
    </div>
  );
}

export default function EvaluationPanel() {
  const [data, setData]       = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [segment, setSegment] = useState<string>("all");

  async function load(seg: string) {
    setLoading(true);
    setError(null);
    try {
      const result = await getEvaluation(seg === "all" ? undefined : seg);
      setData(result);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Failed to load evaluation");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(segment); }, [segment]);

  if (loading) return (
    <div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>
      Computing ML metrics against ground truth…
    </div>
  );

  if (error) return (
    <div style={{ padding: 24, color: "#f87171", fontSize: 13 }}>
      ⚠ {error}
    </div>
  );

  if (!data) return null;

  const { summary, confusion_matrix: cm, metrics, per_segment, threshold_analysis, score_distribution } = data;

  // Precision-Recall curve data
  const prCurve = threshold_analysis.map(t => ({
    threshold: t.threshold,
    precision: t.precision,
    recall: t.recall,
    f1: t.f1,
  }));

  // Score distribution chart
  const distData = score_distribution.map(b => ({
    range: b.range,
    fraud: b.fraud,
    legit: b.legit,
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* ── Segment filter ──────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 11, color: "#64748b" }}>Segment:</span>
        {["all", "VIP", "Regular", "New"].map(s => (
          <button key={s} onClick={() => setSegment(s)} style={{
            fontSize: 11, padding: "3px 12px", borderRadius: 4,
            border: `1px solid ${segment === s ? SEG_COLORS[s] ?? "#3b82f6" : "#1e293b"}`,
            background: segment === s ? (SEG_COLORS[s] ?? "#3b82f6") + "22" : "#0f172a",
            color: segment === s ? (SEG_COLORS[s] ?? "#60a5fa") : "#64748b",
            cursor: "pointer",
          }}>{s}</button>
        ))}
        <button onClick={() => load(segment)} style={{
          fontSize: 11, padding: "3px 10px", borderRadius: 4,
          border: "1px solid #1e293b", background: "#0f172a", color: "#64748b",
          cursor: "pointer", marginLeft: 4,
        }}>↻</button>
        <span style={{ marginLeft: "auto", fontSize: 11, color: "#64748b" }}>
          {summary.total_evaluated.toLocaleString()} predictions evaluated
          · {summary.actual_fraud.toLocaleString()} actual fraud ({(summary.fraud_rate * 100).toFixed(1)}%)
        </span>
      </div>

      {/* Active threshold indicator */}
      <div style={{
        padding: "6px 12px", borderRadius: 6, fontSize: 11,
        background: data.threshold_source === "override" ? "#052e16" : "#0c1a3a",
        border: `1px solid ${data.threshold_source === "override" ? "#16a34a" : "#1e3a5f"}`,
        color: data.threshold_source === "override" ? "#4ade80" : "#60a5fa",
      }}>
        {data.threshold_source === "override"
          ? `⚡ Evaluating at optimized threshold: ${data.active_threshold.toFixed(2)} — metrics reflect the new threshold`
          : `📊 Evaluating at default threshold: ${data.active_threshold.toFixed(2)} — go to ⚡ Threshold Optimizer to improve profit`}
      </div>

      {/* ── KPI row ─────────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10 }}>
        <MetricCard label="AUC-ROC"   value={metrics.auc != null ? metrics.auc.toFixed(4) : "—"}
          good={metrics.auc != null && metrics.auc > 0.9} />
        <MetricCard label="Precision" value={(metrics.precision * 100).toFixed(1) + "%"}
          sub="of BLOCKs are real fraud"
          good={metrics.precision > 0.8} bad={metrics.precision < 0.5} />
        <MetricCard label="Recall"    value={(metrics.recall * 100).toFixed(1) + "%"}
          sub="of fraud caught (BLOCK)"
          good={metrics.recall > 0.6} bad={metrics.recall < 0.3} />
        <MetricCard label="F1 Score"  value={metrics.f1.toFixed(4)}
          good={metrics.f1 > 0.65} bad={metrics.f1 < 0.4} />
        <MetricCard label="Accuracy"  value={(metrics.accuracy * 100).toFixed(2) + "%"} />
        <MetricCard label="False Pos Rate" value={(metrics.fpr * 100).toFixed(2) + "%"}
          sub="legit txns wrongly blocked"
          good={metrics.fpr < 0.005} bad={metrics.fpr > 0.02} />
      </div>

      {/* ── Confusion matrix + Per-segment ──────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Confusion Matrix
          </div>
          <ConfusionMatrix {...cm} />
          <div style={{ marginTop: 12, fontSize: 11, color: "#64748b", display: "flex", gap: 16, justifyContent: "center" }}>
            <span>Precision = TP/(TP+FP)</span>
            <span>Recall = TP/(TP+FN)</span>
          </div>
        </div>

        <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Per-Segment Breakdown
          </div>
          {Object.entries(per_segment).length === 0 ? (
            <div style={{ color: "#64748b", fontSize: 12, textAlign: "center", padding: 20 }}>
              No segment data available
            </div>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
                  <th style={{ textAlign: "left", padding: "4px 8px" }}>Segment</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>N</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>Fraud</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>Prec</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>Recall</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>F1</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(per_segment).map(([seg, m]) => (
                  <tr key={seg} style={{ borderBottom: "1px solid #1e293b" }}>
                    <td style={{ padding: "6px 8px" }}>
                      <span style={{
                        fontSize: 10, padding: "1px 6px", borderRadius: 4,
                        background: (SEG_COLORS[seg] ?? "#3b82f6") + "22",
                        color: SEG_COLORS[seg] ?? "#60a5fa",
                        fontWeight: 600,
                      }}>{seg}</span>
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: "#94a3b8" }}>{m.n.toLocaleString()}</td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: "#f87171" }}>{m.n_fraud.toLocaleString()}</td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: m.precision > 0.8 ? "#4ade80" : "#f59e0b" }}>
                      {(m.precision * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: m.recall > 0.5 ? "#4ade80" : "#f87171" }}>
                      {(m.recall * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: "#e2e8f0", fontWeight: 600 }}>
                      {m.f1.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* ── Threshold analysis + Score distribution ──────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Precision / Recall vs Threshold
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={prCurve}>
              <XAxis dataKey="threshold" tick={{ fill: "#64748b", fontSize: 10 }}
                tickFormatter={v => v.toFixed(1)} axisLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} domain={[0, 1]}
                tickFormatter={v => (v * 100).toFixed(0) + "%"} axisLine={false} />
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "none", borderRadius: 6 }}
                formatter={(v: unknown) => [typeof v === "number" ? (v * 100).toFixed(1) + "%" : String(v)]}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              <Line type="monotone" dataKey="precision" stroke="#3b82f6" strokeWidth={2} dot={false} name="Precision" />
              <Line type="monotone" dataKey="recall"    stroke="#ef4444" strokeWidth={2} dot={false} name="Recall" />
              <Line type="monotone" dataKey="f1"        stroke="#f59e0b" strokeWidth={2} dot={false} name="F1" />
              <ReferenceLine x={0.88} stroke="#6b7280" strokeDasharray="4 2" label={{ value: "Regular", fill: "#6b7280", fontSize: 9 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Fraud Score Distribution (ground truth)
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={distData} margin={{ left: 0, right: 8 }}>
              <XAxis dataKey="range" tick={{ fill: "#64748b", fontSize: 9 }} axisLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} />
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "none", borderRadius: 6 }}
                formatter={(v: unknown) => [typeof v === "number" ? v.toLocaleString() : String(v)]}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              <Bar dataKey="legit" name="Legit"  stackId="a" fill="#1e3a5f" radius={[0, 0, 0, 0]} />
              <Bar dataKey="fraud" name="Fraud"  stackId="a" fill="#ef4444" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Threshold table ──────────────────────────────────────────────── */}
      <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Threshold Analysis — Precision / Recall / F1 at each cutoff
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
                <th style={{ textAlign: "left", padding: "4px 12px" }}>Threshold</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Flagged</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Precision</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Recall</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>F1</th>
                <th style={{ padding: "4px 12px" }}>Bar</th>
              </tr>
            </thead>
            <tbody>
              {threshold_analysis.map(t => {
                const isActive = Math.abs(t.threshold - 0.88) < 0.05;
                return (
                  <tr key={t.threshold} style={{
                    borderBottom: "1px solid #1e293b",
                    background: isActive ? "#1e293b" : "transparent",
                  }}>
                    <td style={{ padding: "5px 12px", color: isActive ? "#f59e0b" : "#94a3b8", fontWeight: isActive ? 700 : 400 }}>
                      {t.threshold.toFixed(2)} {isActive && <span style={{ fontSize: 9, color: "#f59e0b" }}>← current</span>}
                    </td>
                    <td style={{ padding: "5px 12px", textAlign: "right", color: "#94a3b8" }}>{t.flagged.toLocaleString()}</td>
                    <td style={{ padding: "5px 12px", textAlign: "right", color: t.precision > 0.8 ? "#4ade80" : "#f59e0b" }}>
                      {(t.precision * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "5px 12px", textAlign: "right", color: t.recall > 0.5 ? "#4ade80" : "#f87171" }}>
                      {(t.recall * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "5px 12px", textAlign: "right", color: "#e2e8f0", fontWeight: 600 }}>
                      {t.f1.toFixed(3)}
                    </td>
                    <td style={{ padding: "5px 12px" }}>
                      <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
                        <div style={{ height: 6, width: `${t.precision * 60}px`, background: "#3b82f6", borderRadius: 2 }} />
                        <div style={{ height: 6, width: `${t.recall * 60}px`, background: "#ef4444", borderRadius: 2 }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
