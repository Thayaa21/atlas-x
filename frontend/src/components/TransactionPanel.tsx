import React, { useState } from "react";

type AnyObj = Record<string, any>;

export default function TransactionPanel(props: {
  txJson: string;
  txParsed: AnyObj | null;
  onTxJsonChange: (v: string) => void;
}) {
  const { txJson, txParsed, onTxJsonChange } = props;

  const [predictRes, setPredictRes] = useState<any>(null);
  const [dqnRes, setDqnRes] = useState<any>(null);
  const [shapPng, setShapPng] = useState<string>("");
  const [busy, setBusy] = useState<string>("");
  const [error, setError] = useState<string>("");

  async function postJson(url: string) {
    if (!txParsed) throw new Error("Invalid transaction JSON");
    setBusy(url);
    setError("");
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(txParsed),
      });
      const txt = await res.text();
      if (!res.ok) throw new Error(txt || `HTTP ${res.status}`);
      return txt ? JSON.parse(txt) : null;
    } finally {
      setBusy("");
    }
  }

  async function runPredict() {
    setShapPng("");
    setDqnRes(null);
    setPredictRes(null);
    try {
      const data = await postJson("/api/predict");
      setPredictRes(data);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  }

  async function runDqn() {
    setPredictRes(null);
    setShapPng("");
    setDqnRes(null);
    try {
      const data = await postJson("/api/dqn/action");
      setDqnRes(data);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  }

  async function runShap() {
    setPredictRes(null);
    setDqnRes(null);
    setShapPng("");
    try {
      const data = await postJson("/api/shap");
      setShapPng(data?.waterfall_png_base64 ?? "");
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8 }}>
      <h3 style={{ marginTop: 0 }}>Transaction Playground</h3>
      <div style={{ fontSize: 12, color: "#666", marginBottom: 8 }}>
        Paste the raw transaction JSON (the API feature pipeline will compute full features).
      </div>
      <textarea
        value={txJson}
        onChange={(e) => onTxJsonChange(e.target.value)}
        rows={16}
        style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
      />

      <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
        <button onClick={runPredict} disabled={busy !== ""}>
          {busy === "/api/predict" ? "Predict..." : "Predict"}
        </button>
        <button onClick={runDqn} disabled={busy !== ""}>
          {busy === "/api/dqn/action" ? "DQN..." : "DQN Action"}
        </button>
        <button onClick={runShap} disabled={busy !== ""}>
          {busy === "/api/shap" ? "SHAP..." : "SHAP Waterfall"}
        </button>
      </div>

      {error ? (
        <div style={{ marginTop: 10, color: "crimson" }}>
          <b>Error:</b> {error}
        </div>
      ) : null}

      {predictRes ? (
        <pre style={{ marginTop: 10, whiteSpace: "pre-wrap" }}>{JSON.stringify(predictRes, null, 2)}</pre>
      ) : null}

      {dqnRes ? (
        <pre style={{ marginTop: 10, whiteSpace: "pre-wrap" }}>{JSON.stringify(dqnRes, null, 2)}</pre>
      ) : null}

      {shapPng ? (
        <div style={{ marginTop: 12 }}>
          <img
            alt="SHAP waterfall"
            src={`data:image/png;base64,${shapPng}`}
            style={{ width: "100%", borderRadius: 8 }}
          />
        </div>
      ) : null}
    </div>
  );
}

