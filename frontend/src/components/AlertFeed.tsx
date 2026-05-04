import React, { useEffect, useRef, useState } from "react";
import { getRecent } from "../api/client";
import type { RecentTxn } from "../api/client";

interface AlertEntry extends RecentTxn {
  _key: string;
}

interface AlertFeedProps {
  onAlertClick?: (txn: RecentTxn) => void;  // ADD THIS
}

export default function AlertFeed({ onAlertClick }: AlertFeedProps) {  // ADD PROP
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [wsStatus, setWsStatus] = useState<"connecting" | "open" | "polling" | "closed">("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const seenRef = useRef<Set<string>>(new Set());
  const pollRef = useRef<number | null>(null);

  const addAlerts = (items: RecentTxn[]) => {
    const fresh = items.filter(
      (t) => (t.decision === "FLAG" || t.decision === "BLOCK") && !seenRef.current.has(t.transaction_id)
    );
    if (fresh.length === 0) return;
    fresh.forEach((t) => seenRef.current.add(t.transaction_id));
    setAlerts((prev) =>
      [
        ...fresh.map((t) => ({ ...t, _key: `${t.transaction_id}-${Date.now()}` })),
        ...prev,
      ].slice(0, 20)
    );
  };

  const startPolling = () => {
    setWsStatus("polling");
    const poll = async () => {
      try {
        const txns = await getRecent();
        addAlerts(txns);
      } catch {
        // silently retry
      }
      pollRef.current = window.setTimeout(poll, 2000);
    };
    poll();
  };

  useEffect(() => {
    // Always poll HTTP — WS is bonus real-time on top
    startPolling();

    const host = window.location.hostname;
    try {
      const ws = new WebSocket(`ws://${host}:8001/ws/alerts`);
      wsRef.current = ws;
      ws.onopen  = () => setWsStatus("open");
      ws.onclose = () => setWsStatus("polling");
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        try { addAlerts([JSON.parse(ev.data)]); } catch { /* ignore */ }
      };
    } catch { /* WS unavailable — polling covers it */ }

    return () => {
      wsRef.current?.close();
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, []);

  const statusColor = {
    open:       "#22c55e",
    polling:    "#f59e0b",
    connecting: "#9ca3af",
    closed:     "#ef4444",
  }[wsStatus];

  return (
    <div className="panel alert-panel">
      <h2 className="panel-title">
        Alert Feed
        <span className="ws-status" style={{ background: statusColor }} title={wsStatus} />
        <span className="dim" style={{ fontSize: 12 }}>{wsStatus}</span>
      </h2>

      <div className="alert-list">
        {alerts.length === 0 && (
          <div className="empty-state">No alerts yet — waiting for FLAG / BLOCK transactions…</div>
        )}
        {alerts.map((a) => (
          <div 
            key={a._key} 
            className={`alert-row alert-row--${a.decision.toLowerCase()}`}
            onClick={() => onAlertClick?.(a)}  // ADD THIS
            style={{ cursor: 'pointer' }}      // ADD THIS
          >
            <div className="alert-icon">
              {a.decision === "BLOCK" ? "🛑" : "⚠️"}
            </div>
            <div className="alert-body">
              <div className="alert-title">
                <span className={`badge badge--${a.decision.toLowerCase()}`}>{a.decision}</span>
                <span className="mono" style={{ fontSize: 12 }}>{a.transaction_id}</span>
              </div>
              <div className="alert-meta">
                prob <strong>{a.fraud_probability.toFixed(4)}</strong>
                {" · "}graph <strong>{a.graph_risk_score.toFixed(4)}</strong>
                {" · "}seg <strong>{a.customer_segment}</strong>
                {" · "}{a.latency_ms.toFixed(1)} ms
              </div>
            </div>
            <div className="alert-time dim">{new Date(a.timestamp).toLocaleTimeString()}</div>
          </div>
        ))}
      </div>
    </div>
  );
}