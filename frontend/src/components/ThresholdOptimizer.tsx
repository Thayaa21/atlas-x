import React, { useEffect, useState, useCallback } from "react";
import {
  ComposedChart, Line, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend, Area,
} from "recharts";
import {
  optimizeThreshold, applyThreshold, resetThreshold, getCurrentThreshold,
} from "../api/client";
import type { OptimizeResponse, ThresholdCurvePoint } from "../api/client";

function usd(n: number) {
  const abs = Math.abs(n);
  const s = abs >= 1_000_000
    ? "$" + (abs / 1_000_000).toFixed(2) + "M"
    : abs >= 1_000
    ? "$" + (abs / 1_000).toFixed(1) + "k"
    : "$" + abs.toFixed(0);
  return (n < 0 ? "-" : "+") + s;
}

function pct(n: number) { return (n * 100).toFixed(1) + "%"; }

export default function ThresholdOptimizer({ onApplied }: { onApplied?: () => void }) {
  const [data, setData]           = useState<OptimizeResponse | null>(null);
  const [loading, setLoading]     = useState(true);
  const [applying, setApplying]   = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [applied, setApplied]     = useState(false);
  const [sliderVal, setSliderVal] = useState<number | null>(null);
  const [currentInfo, setCurrentInfo] = useState<{
    override_active: boolean; override_value: number | null;
    effective: Record<string, number>;
  } | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [opt, cur] = await Promise.all([
        optimizeThreshold(),
        getCurrentThreshold(),
      ]);
      setData(opt);
      setCurrentInfo(cur);
      setSliderVal(opt.optimal_threshold);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Get the curve point for the current slider value
  const sliderPoint: ThresholdCurvePoint | null = data && sliderVal != null
    ? data.curve.reduce((best, p) =>
        Math.abs(p.threshold - sliderVal) < Math.abs(best.threshold - sliderVal) ? p : best
      )
    : null;

  async function handleApply() {
    if (!sliderVal) return;
    setApplying(true);
    setStatusMsg(null);
    try {
      await applyThreshold(sliderVal);
      setApplied(true);
      setStatusMsg(`✅ Threshold ${sliderVal.toFixed(2)} applied. Switching to Financial Impact…`);
      const cur = await getCurrentThreshold();
      setCurrentInfo(cur);
      // Auto-navigate to Financial Impact tab after a short delay
      setTimeout(() => { onApplied?.(); }, 800);
    } catch (e: any) {
      setStatusMsg(`❌ Failed: ${e?.response?.data?.detail ?? e?.message}`);
    } finally {
      setApplying(false);
    }
  }

  async function handleReset() {
    setResetting(true);
    setStatusMsg(null);
    try {
      const res = await resetThreshold();
      setApplied(false);
      setStatusMsg(`↩ Reset to per-segment defaults: VIP=${res.thresholds.VIP}, Regular=${res.thresholds.Regular}, New=${res.thresholds.New}`);
      const cur = await getCurrentThreshold();
      setCurrentInfo(cur);
      if (data) setSliderVal(data.optimal_threshold);
    } catch (e: any) {
      setStatusMsg(`❌ Failed: ${e?.response?.data?.detail ?? e?.message}`);
    } finally {
      setResetting(false);
    }
  }

  if (loading) return (
    <div style={{ padding: 60, textAlign: "center", color: "#64748b" }}>
      Scanning 91 thresholds (0.05–0.95) for maximum profit…
    </div>
  );

  if (error) return (
    <div style={{ padding: 24, color: "#f87171", fontSize: 13 }}>⚠ {error}</div>
  );

  if (!data) return null;

  const improvement = data.improvement;
  const isOptimalApplied = currentInfo?.override_active &&
    Math.abs((currentInfo.override_value ?? 0) - data.optimal_threshold) < 0.005;

  // Chart data — net impact curve + reference lines
  const chartData = data.curve.map(p => ({
    t:      p.threshold,
    net:    Math.round(p.net_impact),
    recall: Math.round(p.recall * 100),
    prec:   Math.round(p.precision * 100),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div>
        <div style={{ fontSize: 16, fontWeight: 700, color: "#f1f5f9", marginBottom: 4 }}>
          Threshold Optimizer — Maximum Profit
        </div>
        <div style={{ fontSize: 12, color: "#64748b" }}>
          Scanned {data.n_evaluated.toLocaleString()} labeled predictions with real transaction amounts.
          Financial model: saved = TP × 4.60, lost = FN × 4.60, FP cost = FP × 1.63.
        </div>
      </div>

      {/* ── Current vs Optimal summary ───────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
        <div style={{
          background: "#2d0a0a", border: "2px solid #dc2626",
          borderRadius: 10, padding: "16px 20px",
        }}>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 6 }}>
            Current Threshold
            {currentInfo?.override_active && (
              <span style={{ marginLeft: 6, color: "#f59e0b", fontSize: 9 }}>OVERRIDE ACTIVE</span>
            )}
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#f87171" }}>
            {data.current_threshold.toFixed(2)}
          </div>
          <div style={{ fontSize: 13, color: "#f87171", marginTop: 4 }}>
            {usd(data.current_net_impact)} net impact
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            VIP={currentInfo?.effective.VIP?.toFixed(2) ?? "—"}
            · Reg={currentInfo?.effective.Regular?.toFixed(2) ?? "—"}
            · New={currentInfo?.effective.New?.toFixed(2) ?? "—"}
          </div>
        </div>

        <div style={{
          background: "#052e16", border: "2px solid #16a34a",
          borderRadius: 10, padding: "16px 20px",
        }}>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 6 }}>
            Optimal Threshold
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#4ade80" }}>
            {data.optimal_threshold.toFixed(2)}
          </div>
          <div style={{ fontSize: 13, color: "#4ade80", marginTop: 4 }}>
            {usd(data.optimal_net_impact)} net impact
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            Maximises: saved − lost − FP cost − review cost
          </div>
        </div>

        <div style={{
          background: "#0c1a3a", border: "2px solid #3b82f6",
          borderRadius: 10, padding: "16px 20px",
        }}>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 6 }}>
            Potential Improvement
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#60a5fa" }}>
            {usd(improvement)}
          </div>
          <div style={{ fontSize: 13, color: "#60a5fa", marginTop: 4 }}>
            by switching from {data.current_threshold.toFixed(2)} → {data.optimal_threshold.toFixed(2)}
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            {((improvement / Math.abs(data.current_net_impact)) * 100).toFixed(0)}% improvement
          </div>
        </div>
      </div>

      {/* ── Net impact curve ─────────────────────────────────────────────── */}
      <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Net Financial Impact vs Threshold (real transaction amounts)
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={chartData} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
            <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 10 }}
              tickFormatter={v => v.toFixed(2)} axisLine={false} />
            <YAxis yAxisId="net" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false}
              tickFormatter={v => v >= 0 ? "+$" + (v/1000).toFixed(0)+"k" : "-$" + (Math.abs(v)/1000).toFixed(0)+"k"} />
            <YAxis yAxisId="pct" orientation="right" tick={{ fill: "#64748b", fontSize: 10 }}
              axisLine={false} tickFormatter={v => v + "%"} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "none", borderRadius: 6, fontSize: 11 }}
              formatter={(v: unknown) => {
                if (typeof v === "number") return ["$" + v.toLocaleString()];
                return [String(v) + "%"];
              }}
              labelFormatter={v => `Threshold: ${Number(v).toFixed(2)}`}
            />
            <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
            <ReferenceLine yAxisId="net" y={0} stroke="#374151" strokeDasharray="4 2" />
            {/* Current threshold */}
            <ReferenceLine yAxisId="net" x={data.current_threshold}
              stroke="#ef4444" strokeDasharray="4 2"
              label={{ value: `Current ${data.current_threshold.toFixed(2)}`, fill: "#ef4444", fontSize: 9, position: "top" }} />
            {/* Optimal threshold */}
            <ReferenceLine yAxisId="net" x={data.optimal_threshold}
              stroke="#4ade80" strokeWidth={2}
              label={{ value: `Optimal ${data.optimal_threshold.toFixed(2)}`, fill: "#4ade80", fontSize: 9, position: "top" }} />
            {/* Slider position */}
            {sliderVal != null && Math.abs(sliderVal - data.optimal_threshold) > 0.01 && (
              <ReferenceLine yAxisId="net" x={sliderVal}
                stroke="#f59e0b" strokeDasharray="3 2"
                label={{ value: `Preview ${sliderVal.toFixed(2)}`, fill: "#f59e0b", fontSize: 9, position: "insideTopRight" }} />
            )}
            <Area yAxisId="net" type="monotone" dataKey="net" name="Net Impact"
              stroke="#3b82f6" fill="#1e3a5f" strokeWidth={2} dot={false} />
            <Line yAxisId="pct" type="monotone" dataKey="recall" name="Recall %"
              stroke="#ef4444" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
            <Line yAxisId="pct" type="monotone" dataKey="prec" name="Precision %"
              stroke="#22c55e" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Interactive slider ───────────────────────────────────────────── */}
      <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 20 }}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 16, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Preview & Apply Threshold
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
          <span style={{ fontSize: 11, color: "#64748b", minWidth: 60 }}>0.05</span>
          <input
            type="range" min={5} max={95} step={1}
            value={sliderVal != null ? Math.round(sliderVal * 100) : 53}
            onChange={e => setSliderVal(parseInt(e.target.value) / 100)}
            style={{ flex: 1, accentColor: "#3b82f6", cursor: "pointer" }}
          />
          <span style={{ fontSize: 11, color: "#64748b", minWidth: 30 }}>0.95</span>
          <div style={{
            fontSize: 24, fontWeight: 800, color: "#f1f5f9",
            minWidth: 60, textAlign: "center",
          }}>
            {sliderVal?.toFixed(2) ?? "—"}
          </div>
        </div>

        {/* Live preview of selected threshold */}
        {sliderPoint && (
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10,
            marginBottom: 16,
          }}>
            {[
              { label: "Net Impact",  value: usd(sliderPoint.net_impact),
                color: sliderPoint.net_impact >= 0 ? "#4ade80" : "#f87171" },
              { label: "Recall",      value: pct(sliderPoint.recall),      color: "#f87171" },
              { label: "Precision",   value: pct(sliderPoint.precision),   color: "#22c55e" },
              { label: "F1",          value: sliderPoint.f1.toFixed(3),    color: "#f59e0b" },
              { label: "Fraud Caught",value: sliderPoint.tp.toLocaleString(), color: "#4ade80" },
              { label: "Fraud Missed",value: sliderPoint.fn.toLocaleString(), color: "#f87171" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: "#0f172a", border: "1px solid #1e293b",
                borderRadius: 6, padding: "10px 12px", textAlign: "center",
              }}>
                <div style={{ fontSize: 9, color: "#64748b", marginBottom: 4, textTransform: "uppercase" }}>{label}</div>
                <div style={{ fontSize: 16, fontWeight: 700, color }}>{value}</div>
              </div>
            ))}
          </div>
        )}

        {/* Quick-select buttons */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
          <span style={{ fontSize: 11, color: "#64748b", alignSelf: "center" }}>Quick select:</span>
          {[
            { label: `Optimal (${data.optimal_threshold.toFixed(2)})`, t: data.optimal_threshold, color: "#4ade80" },
            { label: "Current (0.88)", t: 0.88, color: "#f87171" },
            { label: "Balanced (0.60)", t: 0.60, color: "#f59e0b" },
            { label: "High Recall (0.30)", t: 0.30, color: "#60a5fa" },
            { label: "High Precision (0.80)", t: 0.80, color: "#a78bfa" },
          ].map(({ label, t, color }) => (
            <button key={label} onClick={() => setSliderVal(t)} style={{
              fontSize: 11, padding: "4px 12px", borderRadius: 4,
              border: `1px solid ${color}44`,
              background: sliderVal != null && Math.abs(sliderVal - t) < 0.005 ? color + "22" : "#0f172a",
              color,
              cursor: "pointer",
            }}>
              {label}
            </button>
          ))}
        </div>

        {/* Apply / Reset buttons */}
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <button
            onClick={handleApply}
            disabled={applying || sliderVal == null}
            style={{
              fontSize: 13, fontWeight: 700, padding: "10px 28px", borderRadius: 6,
              border: "none",
              background: applying ? "#1e293b" : "#16a34a",
              color: applying ? "#64748b" : "#fff",
              cursor: applying ? "not-allowed" : "pointer",
              transition: "background 0.15s",
            }}
          >
            {applying ? "Applying…" : `⚡ Apply Threshold ${sliderVal?.toFixed(2) ?? ""}`}
          </button>

          <button
            onClick={handleReset}
            disabled={resetting || !currentInfo?.override_active}
            style={{
              fontSize: 13, fontWeight: 600, padding: "10px 20px", borderRadius: 6,
              border: "1px solid #374151",
              background: "#1e293b",
              color: currentInfo?.override_active ? "#e2e8f0" : "#475569",
              cursor: (resetting || !currentInfo?.override_active) ? "not-allowed" : "pointer",
            }}
          >
            {resetting ? "Resetting…" : "↩ Reset to Defaults"}
          </button>

          <button onClick={load} style={{
            fontSize: 11, padding: "10px 16px", borderRadius: 6,
            border: "1px solid #1e293b", background: "#0f172a", color: "#64748b",
            cursor: "pointer",
          }}>
            ↻ Re-scan
          </button>
        </div>

        {statusMsg && (
          <div style={{
            marginTop: 12, fontSize: 12, padding: "8px 12px", borderRadius: 6,
            background: statusMsg.startsWith("✅") ? "#052e16" : statusMsg.startsWith("↩") ? "#0c1a3a" : "#2d0a0a",
            color: statusMsg.startsWith("✅") ? "#4ade80" : statusMsg.startsWith("↩") ? "#60a5fa" : "#f87171",
            border: `1px solid ${statusMsg.startsWith("✅") ? "#16a34a" : statusMsg.startsWith("↩") ? "#3b82f6" : "#dc2626"}`,
          }}>
            {statusMsg}
          </div>
        )}
      </div>

      {/* ── Top 10 table ─────────────────────────────────────────────────── */}
      <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Top 10 Thresholds by Net Financial Impact
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
              <th style={{ textAlign: "left", padding: "4px 12px" }}>Threshold</th>
              <th style={{ textAlign: "right", padding: "4px 12px" }}>Net Impact</th>
              <th style={{ textAlign: "right", padding: "4px 12px" }}>Recall</th>
              <th style={{ textAlign: "right", padding: "4px 12px" }}>Precision</th>
              <th style={{ textAlign: "right", padding: "4px 12px" }}>F1</th>
              <th style={{ textAlign: "right", padding: "4px 12px" }}>Fraud Caught</th>
              <th style={{ textAlign: "right", padding: "4px 12px" }}>Fraud Missed</th>
            </tr>
          </thead>
          <tbody>
            {[...data.curve]
              .sort((a, b) => b.net_impact - a.net_impact)
              .slice(0, 10)
              .map((p, i) => {
                const isOptimal  = Math.abs(p.threshold - data.optimal_threshold) < 0.005;
                const isCurrent  = Math.abs(p.threshold - data.current_threshold) < 0.005;
                const isSelected = sliderVal != null && Math.abs(p.threshold - sliderVal) < 0.005;
                return (
                  <tr key={p.threshold}
                    onClick={() => setSliderVal(p.threshold)}
                    style={{
                      borderBottom: "1px solid #1e293b",
                      background: isSelected ? "#1e293b" : isOptimal ? "#052e16" : "transparent",
                      cursor: "pointer",
                    }}
                  >
                    <td style={{ padding: "6px 12px", fontWeight: isOptimal ? 700 : 400 }}>
                      <span style={{ color: isOptimal ? "#4ade80" : isCurrent ? "#f87171" : "#94a3b8" }}>
                        {p.threshold.toFixed(2)}
                      </span>
                      {isOptimal && <span style={{ marginLeft: 6, fontSize: 9, color: "#4ade80" }}>★ OPTIMAL</span>}
                      {isCurrent && !isOptimal && <span style={{ marginLeft: 6, fontSize: 9, color: "#f87171" }}>← current</span>}
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right",
                      color: p.net_impact >= 0 ? "#4ade80" : "#f87171", fontWeight: 700 }}>
                      {usd(p.net_impact)}
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right", color: "#f87171" }}>
                      {pct(p.recall)}
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right", color: "#22c55e" }}>
                      {pct(p.precision)}
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right", color: "#f59e0b" }}>
                      {p.f1.toFixed(3)}
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right", color: "#4ade80" }}>
                      {p.tp.toLocaleString()}
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right", color: "#f87171" }}>
                      {p.fn.toLocaleString()}
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
        <div style={{ marginTop: 8, fontSize: 10, color: "#475569" }}>
          Click any row to preview that threshold. Net = (TP × 4.60) − (FN × 4.60) − (FP × 1.63) − human review.
          Real TransactionAmt from IEEE-CIS dataset.
        </div>
      </div>

    </div>
  );
}
