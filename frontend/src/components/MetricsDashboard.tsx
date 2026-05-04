import React, { useEffect, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { getStats, getCurrentThreshold } from "../api/client";
import type { Stats } from "../api/client";
import { mockStats } from "../utils/mockData";

const DECISION_COLORS: Record<string, string> = {
  APPROVE: "#22c55e",
  FLAG:    "#f59e0b",
  BLOCK:   "#ef4444",
};

interface Props {
  useMock?:   boolean;
  refreshMs?: number;
}

export default function MetricsDashboard({ useMock = false, refreshMs = 5000 }: Props) {
  const [stats, setStats]           = useState<Stats | null>(null);
  const [threshold, setThreshold]   = useState<number | null>(null);
  const [isOverride, setIsOverride] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function fetch() {
      try {
        const s = useMock ? mockStats() : await getStats();
        if (!cancelled) setStats(s);
      } catch {
        if (!cancelled) setStats(mockStats());
      }
      // Also refresh threshold info
      if (!useMock) {
        try {
          const info = await getCurrentThreshold();
          if (!cancelled) {
            setThreshold(info.override_value ?? info.effective?.Regular ?? 0.88);
            setIsOverride(info.override_active);
          }
        } catch {}
      }
    }

    fetch();
    const id = setInterval(fetch, refreshMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [useMock, refreshMs]);

  if (!stats) {
    return <div className="panel metrics-panel"><span className="dim">Loading…</span></div>;
  }

  const pieData    = Object.entries(stats.decisions).map(([name, value]) => ({ name, value }));
  const fraudPct   = (stats.fraud_rate * 100).toFixed(2);
  const total      = stats.total_predictions || 1;
  const blockPct   = ((stats.decisions.BLOCK ?? 0) / total * 100).toFixed(1);
  const flagPct    = ((stats.decisions.FLAG  ?? 0) / total * 100).toFixed(1);

  return (
    <div className="panel metrics-panel">
      <h2 className="panel-title">System Metrics</h2>

      {/* Threshold indicator */}
      {threshold != null && (
        <div style={{
          marginBottom: 10,
          padding: "5px 10px",
          borderRadius: 5,
          background: isOverride ? "rgba(34,197,94,0.08)" : "rgba(107,114,128,0.08)",
          border: `1px solid ${isOverride ? "#16a34a55" : "#37415155"}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}>
          <span style={{ fontSize: 10, color: "#6b7280" }}>
            Active threshold
          </span>
          <span style={{
            fontSize: 13, fontWeight: 700,
            color: isOverride ? "#4ade80" : "#9ca3af",
          }}>
            {threshold.toFixed(2)}
            {isOverride && (
              <span style={{
                marginLeft: 6, fontSize: 9, color: "#4ade80",
                background: "#052e16", padding: "1px 5px", borderRadius: 3,
              }}>
                OPTIMISED
              </span>
            )}
          </span>
        </div>
      )}

      {/* KPI cards */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Total Scored</div>
          <div className="stat-value">{stats.total_predictions.toLocaleString()}</div>
        </div>
        <div className="stat-card stat-card--warn">
          <div className="stat-label">Fraud Rate</div>
          <div className="stat-value">{fraudPct}%</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Latency</div>
          <div className="stat-value">{stats.avg_latency_ms.toFixed(1)} ms</div>
        </div>
        <div className="stat-card stat-card--danger">
          <div className="stat-label">Blocked</div>
          <div className="stat-value">{(stats.decisions.BLOCK ?? 0).toLocaleString()}</div>
        </div>
      </div>

      {/* Decision breakdown bars */}
      <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 5 }}>
        {[
          { key: "APPROVE", label: "Approve", color: "#22c55e" },
          { key: "FLAG",    label: "Flag",    color: "#f59e0b" },
          { key: "BLOCK",   label: "Block",   color: "#ef4444" },
        ].map(({ key, label, color }) => {
          const count = stats.decisions[key] ?? 0;
          const pct   = (count / total * 100);
          return (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 10, color: "#6b7280", width: 46, textAlign: "right" }}>
                {label}
              </span>
              <div style={{
                flex: 1, height: 6, background: "#1f2937", borderRadius: 3, overflow: "hidden",
              }}>
                <div style={{
                  width: `${pct}%`, height: "100%",
                  background: color, borderRadius: 3,
                  transition: "width 0.4s ease",
                }} />
              </div>
              <span style={{ fontSize: 10, color, width: 38, textAlign: "right" }}>
                {count.toLocaleString()}
              </span>
              <span style={{ fontSize: 9, color: "#4b5563", width: 32 }}>
                {pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* Pie chart */}
      <div className="pie-wrapper">
        <ResponsiveContainer width="100%" height={160}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%" cy="50%"
              innerRadius={40} outerRadius={65}
              paddingAngle={3}
              dataKey="value"
            >
              {pieData.map((entry) => (
                <Cell key={entry.name} fill={DECISION_COLORS[entry.name] ?? "#6b7280"} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: "#2d2d2d", border: "1px solid #444", borderRadius: 6 }}
              labelStyle={{ color: "#e5e7eb" }}
              itemStyle={{ color: "#e5e7eb" }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pie-legend">
          {pieData.map((d) => (
            <span key={d.name} className="legend-item">
              <span className="legend-dot" style={{ background: DECISION_COLORS[d.name] ?? "#6b7280" }} />
              {d.name}: {d.value.toLocaleString()}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
