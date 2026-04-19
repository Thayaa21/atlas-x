import React, { useState } from "react";
import "./app.css";
import MetricsDashboard from "./components/MetricsDashboard";
import TransactionFeed from "./components/TransactionFeed";
import FraudRingGraph from "./components/FraudRingGraph";
import ShapExplainer from "./components/ShapExplainer";
import AlertFeed from "./components/AlertFeed";
import type { RecentTxn } from "./api/client";

// Set USE_MOCK=true to run the UI without a backend
const USE_MOCK = false;

export default function App() {
  const [selectedTxn, setSelectedTxn] = useState<RecentTxn | null>(null);

  return (
    <div className="layout">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-logo">
          <span className="logo-atlas">ATLAS</span>
          <span className="logo-x">-X</span>
        </div>
        <div className="topbar-sub">Real-Time Fraud Detection Dashboard</div>
        {USE_MOCK && <div className="topbar-mock-badge">MOCK DATA</div>}
      </header>

      {/* ── Main grid ───────────────────────────────────────────────── */}
      <main className="grid">
        {/* Row 1: metrics (left) + alert feed (right) */}
        <MetricsDashboard useMock={USE_MOCK} refreshMs={5000} />
        <AlertFeed onAlertClick={setSelectedTxn} />

        {/* Row 2: transaction feed (spans 2 cols) */}
        <div className="col-span-2">
          <TransactionFeed
            useMock={USE_MOCK}
            refreshMs={3000}
            onSelect={setSelectedTxn}
          />
        </div>

        {/* Row 3: fraud ring graph (left) + SHAP explainer (right) */}
        <FraudRingGraph useMock={USE_MOCK} />
        <ShapExplainer selectedTxn={selectedTxn} useMock={USE_MOCK} />
      </main>
    </div>
  );
}
