/**
 * ThresholdBar — compact threshold control for the user dashboard top bar.
 *
 * Shows the active threshold, lets the user apply the optimal one or
 * type a custom value. Changes take effect immediately on the next
 * /predict call — the live transaction feed will show updated BLOCK/FLAG/APPROVE
 * decisions within seconds.
 */
import React, { useEffect, useState, useCallback } from "react";
import {
  getCurrentThreshold,
  applyThreshold,
  resetThreshold,
  optimizeThreshold,
} from "../api/client";

export default function ThresholdBar() {
  const [current, setCurrent]       = useState<number | null>(null);
  const [isOverride, setIsOverride] = useState(false);
  const [optimal, setOptimal]       = useState<number | null>(null);
  const [optNet, setOptNet]         = useState<number | null>(null);
  const [applying, setApplying]     = useState(false);
  const [msg, setMsg]               = useState<string | null>(null);
  const [editing, setEditing]       = useState(false);
  const [inputVal, setInputVal]     = useState("");

  const refresh = useCallback(async () => {
    try {
      const info = await getCurrentThreshold();
      const eff  = info.effective;
      // Show the Regular threshold as the representative value
      const t = info.override_value ?? eff.Regular ?? 0.88;
      setCurrent(t);
      setIsOverride(info.override_active);
    } catch {}
  }, []);

  // Load optimal threshold once
  useEffect(() => {
    refresh();
    optimizeThreshold().then(r => {
      setOptimal(r.optimal_threshold);
      setOptNet(r.optimal_net_impact);
    }).catch(() => {});
  }, [refresh]);

  async function handleApplyOptimal() {
    if (!optimal) return;
    setApplying(true);
    try {
      await applyThreshold(optimal);
      await refresh();
      setMsg(`✅ Optimal threshold ${optimal.toFixed(2)} applied — live scoring updated`);
      setTimeout(() => setMsg(null), 4000);
    } catch (e: any) {
      setMsg(`❌ ${e?.message}`);
    } finally {
      setApplying(false);
    }
  }

  async function handleApplyCustom() {
    const t = parseFloat(inputVal);
    if (isNaN(t) || t < 0.01 || t > 0.99) {
      setMsg("Enter a value between 0.01 and 0.99");
      return;
    }
    setApplying(true);
    try {
      await applyThreshold(t);
      await refresh();
      setEditing(false);
      setMsg(`✅ Threshold ${t.toFixed(2)} applied`);
      setTimeout(() => setMsg(null), 3000);
    } catch (e: any) {
      setMsg(`❌ ${e?.message}`);
    } finally {
      setApplying(false);
    }
  }

  async function handleReset() {
    setApplying(true);
    try {
      await resetThreshold();
      await refresh();
      setMsg("↩ Reset to defaults (VIP=0.72, Regular=0.88, New=0.82)");
      setTimeout(() => setMsg(null), 3000);
    } catch {} finally {
      setApplying(false);
    }
  }

  const isOptimalActive = optimal != null && current != null &&
    Math.abs(current - optimal) < 0.005;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, position: "relative" }}>

      {/* Active threshold pill */}
      <div style={{
        display: "flex", alignItems: "center", gap: 6,
        padding: "4px 10px", borderRadius: 6,
        background: isOverride ? "rgba(34,197,94,0.12)" : "rgba(107,114,128,0.15)",
        border: `1px solid ${isOverride ? "#16a34a" : "#374151"}`,
      }}>
        <span style={{ fontSize: 10, color: isOverride ? "#4ade80" : "#9ca3af" }}>
          {isOverride ? "⚡ THRESHOLD" : "THRESHOLD"}
        </span>
        <span style={{ fontSize: 14, fontWeight: 700, color: isOverride ? "#4ade80" : "#e5e7eb" }}>
          {current != null ? current.toFixed(2) : "—"}
        </span>
        {isOverride && (
          <span style={{ fontSize: 9, color: "#4ade80", background: "#052e16", padding: "1px 4px", borderRadius: 3 }}>
            OVERRIDE
          </span>
        )}
      </div>

      {/* Apply optimal button — only show if not already optimal */}
      {optimal != null && !isOptimalActive && (
        <button
          onClick={handleApplyOptimal}
          disabled={applying}
          title={`Apply optimal threshold ${optimal.toFixed(2)} → net impact +$${optNet != null ? Math.abs(optNet).toLocaleString() : "?"}`}
          style={{
            fontSize: 11, fontWeight: 600, padding: "4px 12px", borderRadius: 6,
            border: "1px solid #16a34a",
            background: applying ? "#1f2937" : "rgba(22,163,74,0.15)",
            color: applying ? "#6b7280" : "#4ade80",
            cursor: applying ? "not-allowed" : "pointer",
            whiteSpace: "nowrap",
          }}
        >
          {applying ? "…" : `⚡ Apply Optimal (${optimal.toFixed(2)})`}
        </button>
      )}

      {/* Already optimal badge */}
      {isOptimalActive && (
        <span style={{
          fontSize: 10, padding: "3px 8px", borderRadius: 4,
          background: "rgba(34,197,94,0.1)", color: "#4ade80",
          border: "1px solid #16a34a",
        }}>
          ★ Optimal active
        </span>
      )}

      {/* Custom input */}
      {editing ? (
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <input
            type="number" min="0.01" max="0.99" step="0.01"
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleApplyCustom()}
            autoFocus
            style={{
              width: 60, fontSize: 12, padding: "3px 6px", borderRadius: 4,
              border: "1px solid #374151", background: "#1f2937", color: "#e5e7eb",
            }}
          />
          <button onClick={handleApplyCustom} disabled={applying} style={{
            fontSize: 11, padding: "3px 8px", borderRadius: 4,
            border: "1px solid #374151", background: "#374151", color: "#e5e7eb",
            cursor: "pointer",
          }}>Apply</button>
          <button onClick={() => setEditing(false)} style={{
            fontSize: 11, padding: "3px 6px", borderRadius: 4,
            border: "1px solid #374151", background: "transparent", color: "#9ca3af",
            cursor: "pointer",
          }}>✕</button>
        </div>
      ) : (
        <button
          onClick={() => { setEditing(true); setInputVal(current?.toFixed(2) ?? "0.53"); }}
          style={{
            fontSize: 11, padding: "4px 8px", borderRadius: 4,
            border: "1px solid #374151", background: "transparent", color: "#6b7280",
            cursor: "pointer",
          }}
          title="Set custom threshold"
        >
          ✎
        </button>
      )}

      {/* Reset button — only when override is active */}
      {isOverride && (
        <button
          onClick={handleReset}
          disabled={applying}
          style={{
            fontSize: 11, padding: "4px 8px", borderRadius: 4,
            border: "1px solid #374151", background: "transparent", color: "#9ca3af",
            cursor: "pointer",
          }}
          title="Reset to per-segment defaults"
        >
          ↩ Reset
        </button>
      )}

      {/* Status message */}
      {msg && (
        <div style={{
          position: "absolute", top: "calc(100% + 6px)", right: 0,
          fontSize: 11, padding: "6px 12px", borderRadius: 6, whiteSpace: "nowrap",
          background: msg.startsWith("✅") ? "#052e16" : msg.startsWith("↩") ? "#0c1a3a" : "#2d0a0a",
          color: msg.startsWith("✅") ? "#4ade80" : msg.startsWith("↩") ? "#60a5fa" : "#f87171",
          border: `1px solid ${msg.startsWith("✅") ? "#16a34a" : msg.startsWith("↩") ? "#3b82f6" : "#dc2626"}`,
          zIndex: 100,
        }}>
          {msg}
        </div>
      )}
    </div>
  );
}
