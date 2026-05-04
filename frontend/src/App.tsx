import React, { useState } from "react";
import "./app.css";
import MetricsDashboard from "./components/MetricsDashboard";
import TransactionFeed from "./components/TransactionFeed";
import FraudRingGraph from "./components/FraudRingGraph";
import ShapExplainer from "./components/ShapExplainer";
import AlertFeed from "./components/AlertFeed";
import BackendDashboard from "./components/BackendDashboard";
import ThresholdBar from "./components/ThresholdBar";
import type { RecentTxn } from "./api/client";

const USE_MOCK = false;

export default function App() {
  const [selectedTxn, setSelectedTxn] = useState<RecentTxn | null>(null);
  const [showBackend, setShowBackend]  = useState(false);

  return (
    <div className="layout">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-logo">
          <span className="logo-atlas">ATLAS</span>
          <span className="logo-x">-X</span>
        </div>
        <div className="topbar-sub">Real-Time Fraud Detection Dashboard</div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
          {USE_MOCK && <div className="topbar-mock-badge">MOCK DATA</div>}

          {/* Live threshold control */}
          {!USE_MOCK && <ThresholdBar />}

          {/* Backend dashboard button */}
          <button
            onClick={() => setShowBackend(true)}
            style={{
              fontSize: 12,
              fontWeight: 600,
              padding: "6px 16px",
              borderRadius: 6,
              border: "1px solid #3b82f6",
              background: "rgba(59,130,246,0.12)",
              color: "#60a5fa",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
              transition: "background 0.15s",
            }}
            onMouseEnter={e => (e.currentTarget.style.background = "rgba(59,130,246,0.25)")}
            onMouseLeave={e => (e.currentTarget.style.background = "rgba(59,130,246,0.12)")}
          >
            ⚙ Backend Dashboard
          </button>
        </div>
      </header>

      {/* ── Main grid ───────────────────────────────────────────────── */}
      <main className="grid">
        <MetricsDashboard useMock={USE_MOCK} refreshMs={3000} />
        <AlertFeed onAlertClick={setSelectedTxn} />

        <div className="col-span-2">
          <TransactionFeed
            useMock={USE_MOCK}
            refreshMs={3000}
            onSelect={setSelectedTxn}
          />
        </div>

        <FraudRingGraph useMock={USE_MOCK} />
        <ShapExplainer selectedTxn={selectedTxn} useMock={USE_MOCK} />
      </main>

      {/* ── Backend Dashboard overlay ────────────────────────────────── */}
      {showBackend && (
        <BackendDashboard onClose={() => setShowBackend(false)} />
      )}
    </div>
  );
}
