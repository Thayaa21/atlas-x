import React, { useEffect, useState, useCallback } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { getFlaggedTransactions, getTransactionEvents } from "../api/client";
import type { FlaggedTransaction, EventRecord } from "../api/client";
import EvaluationPanel from "./EvaluationPanel";
import FinancialImpactPanel from "./FinancialImpactPanel";
import ThresholdOptimizer from "./ThresholdOptimizer";

// ── Types ─────────────────────────────────────────────────────────────────────

interface SystemHealth {
  status: string;
  xgboost_loaded: boolean;
  neo4j_connected: boolean;
  uptime_seconds: number;
}

interface Stats {
  total_predictions: number;
  fraud_rate: number;
  avg_latency_ms: number;
  decisions: Record<string, number>;
}

interface Ring {
  ring_type: string;
  ring_id: string;
  card_count: number;
  fraud_count: number;
  fraud_rate: number;
}

interface LatencyPoint { t: string; ms: number; }

const DECISION_COLORS: Record<string, string> = {
  APPROVE: "#22c55e", FLAG: "#f59e0b", BLOCK: "#ef4444",
};

// ── Helpers ───────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string): Promise<T> {
  const r = await fetch(`http://localhost:8001${path}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function fmt(n: number, dec = 1) { return n.toFixed(dec); }
function pct(n: number) { return (n * 100).toFixed(1) + "%"; }

// ── Main component ────────────────────────────────────────────────────────────

// ── Financial tab wrapper (fetches eval data then renders FinancialImpactPanel) ──

import { getEvaluation } from "../api/client";
import type { EvaluationResponse } from "../api/client";

function FinancialTab({ isActive }: { isActive: boolean }) {
  const [evalData, setEvalData] = React.useState<EvaluationResponse | null>(null);
  const [loading, setLoading]   = React.useState(false);
  const [error, setError]       = React.useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setEvalData(await getEvaluation());
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Failed");
    } finally {
      setLoading(false);
    }
  }

  // Re-fetch every time this tab becomes active (picks up threshold changes)
  React.useEffect(() => {
    if (isActive) load();
  }, [isActive]);

  if (loading || !evalData) return (
    <div style={{ padding: 60, textAlign: "center", color: "#64748b" }}>
      {loading ? "Loading financial data…" : ""}
    </div>
  );
  if (error) return (
    <div style={{ padding: 24, color: "#f87171", fontSize: 13 }}>⚠ {error}</div>
  );

  return (
    <div style={{ padding: "20px 24px" }}>
      {/* Active threshold banner */}
      <div style={{
        marginBottom: 16, padding: "8px 14px", borderRadius: 6,
        background: evalData.threshold_source === "override" ? "#052e16" : "#0c1a3a",
        border: `1px solid ${evalData.threshold_source === "override" ? "#16a34a" : "#1e3a5f"}`,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ fontSize: 12, color: evalData.threshold_source === "override" ? "#4ade80" : "#60a5fa" }}>
          {evalData.threshold_source === "override"
            ? `⚡ Using optimized threshold: ${evalData.active_threshold.toFixed(2)} (override active)`
            : `📊 Using default threshold: ${evalData.active_threshold.toFixed(2)} — go to ⚡ Threshold Optimizer to improve profit`}
        </div>
        <button onClick={load} style={{
          fontSize: 11, padding: "3px 10px", borderRadius: 4,
          border: "1px solid #1e293b", background: "#0f172a", color: "#64748b",
          cursor: "pointer",
        }}>↻ Refresh</button>
      </div>
      <FinancialImpactPanel evalData={evalData} />
    </div>
  );
}

export default function BackendDashboard({ onClose }: { onClose: () => void }) {
  const [health, setHealth]       = useState<SystemHealth | null>(null);
  const [stats, setStats]         = useState<Stats | null>(null);
  const [rings, setRings]         = useState<Ring[]>([]);
  const [flagged, setFlagged]     = useState<FlaggedTransaction[]>([]);
  const [latencyHistory, setLatencyHistory] = useState<LatencyPoint[]>([]);
  const [selectedFlagTxn, setSelectedFlagTxn] = useState<string | null>(null);
  const [events, setEvents]       = useState<EventRecord[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [loading, setLoading]     = useState(true);
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [activeTab, setActiveTab] = useState<"ops" | "evaluation" | "financial" | "threshold">("ops");

  const refresh = useCallback(async () => {
    try {
      const [h, s, r, f] = await Promise.all([
        apiFetch<SystemHealth>("/api/v1/health"),
        apiFetch<Stats>("/api/v1/stats"),
        apiFetch<{ rings: Ring[] }>("/api/v1/graph/rings?min_cards=5&min_frauds=3"),
        getFlaggedTransactions(20),
      ]);
      setHealth(h);
      setStats(s);
      setRings(r.rings.slice(0, 10));
      setFlagged(f.flagged);
      setLatencyHistory(prev => {
        const point = { t: new Date().toLocaleTimeString(), ms: s.avg_latency_ms };
        return [...prev.slice(-19), point];
      });
      setLastRefresh(new Date());
    } catch (e) {
      console.error("Backend dashboard refresh error:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  async function loadEvents(txnId: string) {
    setSelectedFlagTxn(txnId);
    setEventsLoading(true);
    try {
      const h = await getTransactionEvents(txnId);
      setEvents(h.events);
    } catch {
      setEvents([]);
    } finally {
      setEventsLoading(false);
    }
  }

  const decisionPieData = stats
    ? Object.entries(stats.decisions).map(([name, value]) => ({ name, value }))
    : [];

  const uptimeStr = health
    ? `${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor((health.uptime_seconds % 3600) / 60)}m`
    : "—";

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "#0f1117",
      overflowY: "auto",
      fontFamily: "inherit",
    }}>
      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 24px",
        borderBottom: "1px solid #1f2937",
        background: "#111827",
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 18, fontWeight: 700, color: "#f9fafb" }}>
            <span style={{ color: "#3b82f6" }}>ATLAS</span>
            <span style={{ color: "#ef4444" }}>-X</span>
            <span style={{ color: "#6b7280", fontSize: 13, fontWeight: 400, marginLeft: 8 }}>
              Backend Ops Dashboard
            </span>
          </span>
          <span style={{
            fontSize: 10, padding: "2px 8px", borderRadius: 12,
            background: health?.status === "healthy" ? "#14532d" : "#7f1d1d",
            color: health?.status === "healthy" ? "#4ade80" : "#f87171",
            border: `1px solid ${health?.status === "healthy" ? "#16a34a" : "#dc2626"}`,
          }}>
            {health?.status?.toUpperCase() ?? "LOADING"}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {/* Tabs */}
          <div style={{ display: "flex", gap: 4, background: "#0f172a", borderRadius: 6, padding: 3 }}>
            {(["ops", "evaluation", "financial", "threshold"] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} style={{
                fontSize: 12, fontWeight: 600, padding: "5px 16px", borderRadius: 4,
                border: "none",
                background: activeTab === tab ? "#1e293b" : "transparent",
                color: activeTab === tab ? "#f1f5f9" : "#64748b",
                cursor: "pointer",
              }}>
                {tab === "ops" ? "⚙ Ops"
                  : tab === "evaluation" ? "📊 Ground Truth"
                  : tab === "financial" ? "💰 Financial Impact"
                  : "⚡ Threshold Optimizer"}
              </button>
            ))}
          </div>
          <span style={{ fontSize: 11, color: "#6b7280" }}>
            Refreshed {lastRefresh.toLocaleTimeString()}
          </span>
          <button onClick={refresh} style={{
            fontSize: 11, padding: "4px 12px", borderRadius: 4,
            border: "1px solid #374151", background: "#1f2937", color: "#9ca3af",
            cursor: "pointer",
          }}>↻ Refresh</button>
          <button onClick={onClose} style={{
            fontSize: 13, padding: "4px 14px", borderRadius: 4,
            border: "1px solid #374151", background: "#1f2937", color: "#e5e7eb",
            cursor: "pointer", fontWeight: 600,
          }}>✕ Close</button>
        </div>
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 80, color: "#6b7280" }}>
          Loading backend data…
        </div>
      ) : activeTab === "evaluation" ? (
        <div style={{ padding: "20px 24px" }}>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#f1f5f9", marginBottom: 4 }}>
              Ground Truth Evaluation
            </div>
            <div style={{ fontSize: 12, color: "#64748b" }}>
              Comparing XGBoost predictions against actual <code style={{ color: "#60a5fa" }}>isFraud</code> labels
              from the IEEE-CIS dataset holdout. BLOCK decisions are treated as positive predictions.
            </div>
          </div>
          <EvaluationPanel />
        </div>
      ) : activeTab === "financial" ? (
        <FinancialTab isActive={activeTab === "financial"} />
      ) : activeTab === "threshold" ? (
        <div style={{ padding: "20px 24px" }}>
          <ThresholdOptimizer onApplied={() => setActiveTab("financial")} />
        </div>
      ) : (
        <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 20 }}>

          {/* ── ROW 1: System Health KPIs ──────────────────────────────────── */}
          <section>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              System Health
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 12 }}>
              {[
                { label: "API Status",       value: health?.status ?? "—",        ok: health?.status === "healthy" },
                { label: "XGBoost",          value: health?.xgboost_loaded ? "Loaded" : "Not loaded", ok: health?.xgboost_loaded },
                { label: "Neo4j",            value: health?.neo4j_connected ? "Connected" : "Offline", ok: health?.neo4j_connected },
                { label: "Uptime",           value: uptimeStr,                    ok: true },
                { label: "Total Predictions",value: stats?.total_predictions?.toLocaleString() ?? "0", ok: true },
                { label: "Avg Latency",      value: `${fmt(stats?.avg_latency_ms ?? 0)} ms`, ok: (stats?.avg_latency_ms ?? 0) < 50 },
              ].map(({ label, value, ok }) => (
                <div key={label} style={{
                  background: "#111827", border: `1px solid ${ok ? "#1f2937" : "#7f1d1d"}`,
                  borderRadius: 8, padding: "12px 14px",
                }}>
                  <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: ok ? "#f9fafb" : "#f87171" }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* ── ROW 2: Latency chart + Decision pie ───────────────────────── */}
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
            <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                API Latency (avg ms) — live
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={latencyHistory}>
                  <XAxis dataKey="t" tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#1f2937", border: "none", borderRadius: 6 }}
                    labelStyle={{ color: "#9ca3af" }} itemStyle={{ color: "#60a5fa" }} />
                  <Line type="monotone" dataKey="ms" stroke="#3b82f6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                Decision Distribution
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie data={decisionPieData} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" outerRadius={60}
                    label={false}
                    labelLine={false}
                  >
                    {decisionPieData.map((entry) => (
                      <Cell key={entry.name} fill={DECISION_COLORS[entry.name] ?? "#6b7280"} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#1f2937", border: "none", borderRadius: 6 }} />
                </PieChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", justifyContent: "center", gap: 12, marginTop: 4 }}>
                {decisionPieData.map(({ name, value }) => (
                  <div key={name} style={{ fontSize: 11, color: DECISION_COLORS[name] }}>
                    {name}: {value}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── ROW 3: Fraud Rings ─────────────────────────────────────────── */}
          <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Top Fraud Rings (Neo4j) — {rings.length} shown
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={rings} layout="vertical" margin={{ left: 0, right: 40 }}>
                <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} />
                <YAxis type="category" dataKey="ring_id" width={160}
                  tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false}
                  tickFormatter={(v) => v.length > 22 ? v.slice(0, 22) + "…" : v}
                />
                <Tooltip contentStyle={{ background: "#1f2937", border: "none", borderRadius: 6 }}
                  formatter={(v, name) => [v, name === "fraud_count" ? "Fraud cards" : "Total cards"]}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: "#9ca3af" }} />
                <Bar dataKey="card_count"  name="Total cards" fill="#374151" radius={[0, 2, 2, 0]} />
                <Bar dataKey="fraud_count" name="Fraud cards"  fill="#ef4444" radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* ── ROW 4: Human Review Queue + Event Audit ───────────────────── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

            {/* Flagged queue */}
            <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                🚩 Human Review Queue ({flagged.length} flagged)
              </div>
              <div style={{ overflowY: "auto", maxHeight: 280 }}>
                {flagged.length === 0 ? (
                  <div style={{ color: "#6b7280", fontSize: 12, textAlign: "center", padding: 20 }}>
                    No flagged transactions
                  </div>
                ) : (
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead>
                      <tr style={{ color: "#6b7280", borderBottom: "1px solid #1f2937" }}>
                        <th style={{ textAlign: "left", padding: "4px 8px" }}>Txn ID</th>
                        <th style={{ textAlign: "right", padding: "4px 8px" }}>Prob</th>
                        <th style={{ textAlign: "right", padding: "4px 8px" }}>Graph</th>
                        <th style={{ textAlign: "right", padding: "4px 8px" }}>Wait</th>
                        <th style={{ textAlign: "left", padding: "4px 8px" }}>Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {flagged.map((f) => (
                        <tr
                          key={f.transaction_id}
                          onClick={() => loadEvents(f.transaction_id)}
                          style={{
                            borderBottom: "1px solid #1f2937",
                            cursor: "pointer",
                            background: selectedFlagTxn === f.transaction_id ? "#1f2937" : "transparent",
                          }}
                        >
                          <td style={{ padding: "5px 8px", color: "#60a5fa", fontFamily: "monospace" }}>
                            {f.transaction_id}
                          </td>
                          <td style={{ padding: "5px 8px", textAlign: "right", color: "#f87171" }}>
                            {pct(f.fraud_probability)}
                          </td>
                          <td style={{ padding: "5px 8px", textAlign: "right", color: "#f59e0b" }}>
                            {pct(f.graph_risk_score)}
                          </td>
                          <td style={{ padding: "5px 8px", textAlign: "right", color: "#9ca3af" }}>
                            {f.minutes_waiting.toFixed(0)}m
                          </td>
                          <td style={{ padding: "5px 8px", color: "#9ca3af", maxWidth: 140,
                            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {f.flag_reason}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            {/* Event audit trail */}
            <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                📋 Event Audit Trail
                {selectedFlagTxn && (
                  <span style={{ color: "#60a5fa", marginLeft: 8, fontFamily: "monospace" }}>
                    txn {selectedFlagTxn}
                  </span>
                )}
              </div>
              {!selectedFlagTxn && (
                <div style={{ color: "#6b7280", fontSize: 12, textAlign: "center", padding: 20 }}>
                  Click a flagged transaction to see its audit trail
                </div>
              )}
              {eventsLoading && (
                <div style={{ color: "#6b7280", fontSize: 12, textAlign: "center", padding: 20 }}>
                  Loading events…
                </div>
              )}
              {!eventsLoading && selectedFlagTxn && events.length === 0 && (
                <div style={{ color: "#6b7280", fontSize: 12, textAlign: "center", padding: 20 }}>
                  No events found
                </div>
              )}
              {!eventsLoading && events.length > 0 && (
                <div style={{ overflowY: "auto", maxHeight: 280 }}>
                  {events.map((ev, i) => (
                    <div key={i} style={{
                      borderBottom: "1px solid #1f2937", padding: "8px 0",
                      display: "flex", flexDirection: "column", gap: 4,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{
                          fontSize: 10, padding: "1px 6px", borderRadius: 4,
                          background: "#1f2937", color: "#60a5fa", fontWeight: 600,
                        }}>
                          {ev.event_type}
                        </span>
                        <span style={{ fontSize: 10, color: "#6b7280" }}>
                          {new Date(ev.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: "#9ca3af", fontFamily: "monospace" }}>
                        {Object.entries(ev.data)
                          .filter(([k]) => ["decision","fraud_probability","graph_risk_score","customer_segment","latency_ms"].includes(k))
                          .map(([k, v]) => `${k}=${typeof v === "number" ? (v as number).toFixed(3) : v}`)
                          .join("  ·  ")}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* ── ROW 5: Quick links ─────────────────────────────────────────── */}
          <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Quick Links
            </div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {[
                { label: "📊 Grafana Ops Dashboard", url: "http://localhost:3000", color: "#f59e0b" },
                { label: "🔥 Prometheus Metrics",    url: "http://localhost:9090", color: "#ef4444" },
                { label: "🕸 Neo4j Browser",         url: "http://localhost:7474", color: "#22c55e" },
                { label: "📖 API Swagger Docs",      url: "http://localhost:8001/docs", color: "#3b82f6" },
                { label: "📡 Prometheus Scrape",     url: "http://localhost:8001/api/metrics/prometheus", color: "#8b5cf6" },
              ].map(({ label, url, color }) => (
                <a key={url} href={url} target="_blank" rel="noopener noreferrer" style={{
                  fontSize: 12, padding: "6px 14px", borderRadius: 6,
                  border: `1px solid ${color}33`,
                  background: color + "11",
                  color,
                  textDecoration: "none",
                  transition: "background 0.15s",
                }}>
                  {label}
                </a>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
