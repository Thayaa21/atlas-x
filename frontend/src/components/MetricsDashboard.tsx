import React, { useEffect, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { getStats } from "../api/client";
import type { Stats } from "../api/client";
import { mockStats } from "../utils/mockData";

const DECISION_COLORS: Record<string, string> = {
  APPROVE: "#22c55e",
  FLAG: "#f59e0b",
  BLOCK: "#ef4444",
};

interface Props {
  useMock?: boolean;
  refreshMs?: number;
}

export default function MetricsDashboard({ useMock = false, refreshMs = 5000 }: Props) {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetch() {
      try {
        const s = useMock ? mockStats() : await getStats();
        if (!cancelled) setStats(s);
      } catch {
        if (!cancelled) setStats(mockStats());
      }
    }
    fetch();
    const id = setInterval(fetch, refreshMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [useMock, refreshMs]);

  if (!stats) {
    return <div className="panel metrics-panel"><span className="dim">Loading…</span></div>;
  }

  const pieData = Object.entries(stats.decisions).map(([name, value]) => ({ name, value }));
  const fraudPct = (stats.fraud_rate * 100).toFixed(2);

  return (
    <div className="panel metrics-panel">
      <h2 className="panel-title">System Metrics</h2>

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

      <div className="pie-wrapper">
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={45}
              outerRadius={75}
              paddingAngle={3}
              dataKey="value"
            >
              {pieData.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={DECISION_COLORS[entry.name] ?? "#6b7280"}
                />
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
