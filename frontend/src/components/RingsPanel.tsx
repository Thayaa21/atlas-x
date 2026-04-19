import React, { useState } from "react";

type RingRes = {
  ring_id: number;
  pagerank: number;
  // any other fields
  [k: string]: any;
};

export default function RingsPanel() {
  const [nodeType, setNodeType] = useState<string>("Card");
  const [nodeId, setNodeId] = useState<string>("");
  const [ring, setRing] = useState<RingRes | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  async function fetchRings() {
    setLoading(true);
    setError("");
    setRing(null);
    try {
      const url = `/api/graph/rings?node_type=${encodeURIComponent(nodeType)}&node_id=${encodeURIComponent(
        nodeId
      )}`;
      const res = await fetch(url);
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as RingRes;
      setRing(data);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8 }}>
      <h3 style={{ marginTop: 0 }}>Neo4j Fraud Rings</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <label>
          Node type:
          <select value={nodeType} onChange={(e) => setNodeType(e.target.value)} style={{ marginLeft: 8 }}>
            <option value="Card">Card</option>
            <option value="Device">Device</option>
            <option value="IP">IP</option>
            <option value="Email">Email</option>
          </select>
        </label>
      </div>
      <div style={{ marginBottom: 8 }}>
        <label>
          Node id:
          <input
            value={nodeId}
            onChange={(e) => setNodeId(e.target.value)}
            placeholder="e.g. card1-card2-card3"
            style={{ width: "100%", marginTop: 6 }}
          />
        </label>
      </div>
      <button onClick={fetchRings} disabled={loading || nodeId.length === 0}>
        {loading ? "Fetching..." : "Fetch Ring Risk"}
      </button>
      {error ? <div style={{ marginTop: 8, color: "crimson" }}>{error}</div> : null}
      {ring ? (
        <div style={{ marginTop: 12 }}>
          <div>
            <b>ring_id:</b> {ring.ring_id}
          </div>
          <div>
            <b>pagerank:</b> {typeof ring.pagerank === "number" ? ring.pagerank.toFixed(6) : String(ring.pagerank)}
          </div>
        </div>
      ) : null}
      <div style={{ marginTop: 10, fontSize: 12, color: "#666" }}>
        Note: requires `src/graph/fraud_rings.py` output (`node_rings.json`) and FastAPI `/api/graph/rings`.
      </div>
    </div>
  );
}

