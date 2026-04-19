import React, { useEffect, useMemo, useState } from "react";

type AlertRow = {
  TransactionID: number;
  fraud_prob: number;
  segment: string;
  threshold?: number;
  decision: string;
  dqn_action?: string;
  timestamp?: number;
};

export default function WebSocketMonitor() {
  const wsUrl = useMemo(() => {
    const host = window.location.hostname;
    return `ws://${host}:8000/ws/alerts`;
  }, []);

  const [connected, setConnected] = useState(false);
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
    } catch (e: any) {
      setError(String(e?.message ?? e));
      return;
    }

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setError("WebSocket error");

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        setAlerts((prev) => [data as AlertRow, ...prev].slice(0, 50));
      } catch {
        // ignore
      }
    };

    return () => {
      if (ws) ws.close();
    };
  }, [wsUrl]);

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, marginBottom: 12 }}>
      <h3 style={{ marginTop: 0 }}>Real-time Fraud Alerts</h3>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <span>Status: </span>
        <b>{connected ? "Connected" : "Disconnected"}</b>
        {error ? <span style={{ color: "crimson" }}>({error})</span> : null}
      </div>
      <div style={{ marginTop: 12, maxHeight: 360, overflow: "auto" }}>
        {alerts.length === 0 ? (
          <div>No alerts yet.</div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>TxID</th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>Prob</th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>Seg</th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>Decision</th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>DQN</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a) => (
                <tr key={a.TransactionID + String(a.timestamp ?? "")}>
                  <td>{a.TransactionID}</td>
                  <td>{(a.fraud_prob ?? 0).toFixed(4)}</td>
                  <td>{a.segment}</td>
                  <td>{a.decision}</td>
                  <td>{a.dqn_action ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

