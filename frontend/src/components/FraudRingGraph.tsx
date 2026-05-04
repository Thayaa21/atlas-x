import React, { useEffect, useRef, useState } from "react";
import { getFraudRings } from "../api/client";
import type { FraudRing } from "../api/client";

// We build a simple adjacency graph: one hub node per ring, connected to
// synthetic card nodes whose colour encodes fraud vs. clean.

interface GraphNode {
  id: string;
  label: string;
  type: "ring" | "card";
  isFraud?: boolean;
  color: string;
  val: number;
}

interface GraphLink {
  source: string;
  target: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

const RING_COLOR: Record<string, string> = {
  device: "#f59e0b",
  email: "#8b5cf6",
  address: "#3b82f6",
};

function buildGraph(rings: FraudRing[]): GraphData {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];

  for (const ring of rings.slice(0, 8)) {
    const ringId = `ring:${ring.ring_type}:${ring.ring_id}`;
    nodes.push({
      id: ringId,
      label: `${ring.ring_type}\n${ring.ring_id.slice(0, 8)}`,
      type: "ring",
      color: RING_COLOR[ring.ring_type] ?? "#6b7280",
      val: ring.fraud_count * 2,
    });

    for (let i = 0; i < Math.min(ring.card_count, 8); i++) {
      const isFraud = i < ring.fraud_count;
      const cardId = `card:${ring.ring_id}:${i}`;
      nodes.push({
        id: cardId,
        label: `card-${i}`,
        type: "card",
        isFraud,
        color: isFraud ? "#ef4444" : "#374151",
        val: isFraud ? 3 : 1,
      });
      links.push({ source: ringId, target: cardId });
    }
  }

  return { nodes, links };
}

interface Props {
  useMock?: boolean;
}

export default function FraudRingGraph({ useMock = false }: Props) {
  const [rings, setRings] = useState<FraudRing[]>([]);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [hovered, setHovered] = useState<GraphNode | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Lazy-import ForceGraph2D (2D-only package, no aframe dependency)
  const [ForceGraph, setForceGraph] = useState<any>(null);
  useEffect(() => {
    import("react-force-graph-2d").then((m) => setForceGraph(() => m.default));
  }, []);

  useEffect(() => {
  let cancelled = false;
  async function load() {
    try {
      const data = (await getFraudRings()).slice(0, 8);  // Only top 8 rings
      console.log("GOT RINGS:", data.length, data); // ADD THIS
      if (!cancelled) {
        setRings(data);
        setGraphData(buildGraph(data));
      }
    } catch (err) {
      console.error("RING FETCH FAILED:", err); // ADD THIS
    }
  }
  load();
  return () => { cancelled = true; };
}, []);

  const totalFraud = rings.reduce((s, r) => s + r.fraud_count, 0);

  return (
    <div className="panel ring-panel" ref={containerRef}>
      <h2 className="panel-title">
        Fraud Ring Graph
        <span className="panel-badge">{rings.length} rings · {totalFraud} fraud cards</span>
      </h2>

      <div className="ring-legend">
        {Object.entries(RING_COLOR).map(([t, c]) => (
          <span key={t} className="legend-item">
            <span className="legend-dot" style={{ background: c }} />
            {t}
          </span>
        ))}
        <span className="legend-item">
          <span className="legend-dot" style={{ background: "#ef4444" }} />fraud card
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ background: "#374151" }} />clean card
        </span>
      </div>

      {hovered && (
        <div className="ring-tooltip">
          {hovered.type === "ring"
            ? hovered.label
            : hovered.isFraud
            ? "Fraud card"
            : "Clean card"}
        </div>
      )}

      <div className="ring-canvas">
        {ForceGraph && graphData.nodes.length > 0 ? (
          <ForceGraph
            graphData={graphData}
            width={containerRef.current?.clientWidth ?? 400}
            height={280}
            backgroundColor="#1a1a1a"
            nodeColor={(n: GraphNode) => n.color}
            nodeVal={(n: GraphNode) => n.val}
            linkColor={() => "#374151"}
            nodeLabel={(n: GraphNode) => n.label}
            onNodeHover={(n: GraphNode | null) => setHovered(n)}
            cooldownTicks={80}
          />
        ) : (
          <div className="empty-state">Loading graph…</div>
        )}
      </div>

      <div className="ring-table-wrap">
        <table className="ring-table">
          <thead>
            <tr>
              <th>Type</th>
              <th>Ring ID</th>
              <th>Cards</th>
              <th>Fraud</th>
              <th>Rate</th>
            </tr>
          </thead>
          <tbody>
            {rings.slice(0, 10).map((r) => (
              <tr key={`${r.ring_type}-${r.ring_id}`}>
                <td>
                  <span className="legend-dot" style={{ background: RING_COLOR[r.ring_type] ?? "#6b7280", display: "inline-block", marginRight: 4 }} />
                  {r.ring_type}
                </td>
                <td className="mono">{r.ring_id.slice(0, 12)}</td>
                <td>{r.card_count}</td>
                <td style={{ color: "#ef4444" }}>{r.fraud_count}</td>
                <td>{(r.fraud_rate * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
