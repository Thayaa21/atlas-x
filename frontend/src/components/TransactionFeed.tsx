import React, { useEffect, useRef, useState } from "react";
import { getRecent } from "../api/client";
import type { RecentTxn } from "../api/client";
import { mockRecentList, mockRecentTxn } from "../utils/mockData";

const DECISION_CLASS: Record<string, string> = {
  APPROVE: "badge badge--approve",
  FLAG: "badge badge--flag",
  BLOCK: "badge badge--block",
};

interface Props {
  useMock?: boolean;
  refreshMs?: number;
  onSelect?: (txn: RecentTxn) => void;
}

export default function TransactionFeed({ useMock = false, refreshMs = 3000, onSelect }: Props) {
  const [txns, setTxns] = useState<RecentTxn[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const mockInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // In mock mode we inject a new fake transaction every 2s to simulate streaming
  useEffect(() => {
    let cancelled = false;

    async function loadReal() {
      try {
        const data = await getRecent();
        if (!cancelled) setTxns(data);
      } catch {
        if (!cancelled) setTxns(mockRecentList(20));
      }
    }

    if (useMock) {
      setTxns(mockRecentList(20));
      mockInterval.current = setInterval(() => {
        if (cancelled) return;
        setTxns((prev) => [mockRecentTxn(), ...prev].slice(0, 50));
      }, 2000);
    } else {
      loadReal();
      const id = setInterval(loadReal, refreshMs);
      return () => {
        cancelled = true;
        clearInterval(id);
      };
    }

    return () => {
      cancelled = true;
      if (mockInterval.current) clearInterval(mockInterval.current);
    };
  }, [useMock, refreshMs]);

  function handleSelect(txn: RecentTxn) {
    setSelected(txn.transaction_id);
    onSelect?.(txn);
  }

  return (
    <div className="panel txn-panel">
      <h2 className="panel-title">Live Transactions</h2>
      <div className="txn-table-wrap">
        <table className="txn-table">
          <thead>
            <tr>
              <th>TxID</th>
              <th>Decision</th>
              <th>Prob</th>
              <th>Graph Risk</th>
              <th>Segment</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {txns.map((t) => (
              <tr
                key={t.transaction_id + t.timestamp}
                className={`txn-row txn-row--${t.decision.toLowerCase()}${
                  selected === t.transaction_id ? " txn-row--selected" : ""
                }`}
                onClick={() => handleSelect(t)}
              >
                <td className="mono">{t.transaction_id}</td>
                <td>
                  <span className={DECISION_CLASS[t.decision] ?? "badge"}>
                    {t.decision}
                  </span>
                </td>
                <td className="mono">{t.fraud_probability.toFixed(4)}</td>
                <td className="mono">{t.graph_risk_score.toFixed(4)}</td>
                <td>{t.customer_segment}</td>
                <td className="mono">{t.latency_ms.toFixed(1)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
        {txns.length === 0 && <div className="empty-state">Waiting for transactions…</div>}
      </div>
    </div>
  );
}
