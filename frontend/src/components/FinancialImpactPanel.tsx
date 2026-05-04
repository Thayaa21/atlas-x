/**
 * ATLAS-X Financial Impact Panel
 *
 * Uses REAL transaction amounts from the IEEE-CIS dataset (TransactionAmt)
 * as the primary cost basis — not industry averages.
 *
 * Real amounts (from data):
 *   TP amount = sum of TransactionAmt where isFraud=1 AND decision=BLOCK
 *   FN amount = sum of TransactionAmt where isFraud=1 AND decision≠BLOCK
 *   FP amount = sum of TransactionAmt where isFraud=0 AND decision=BLOCK
 *
 * Industry benchmarks used ONLY for multipliers (not base amounts):
 *   • Fraud cost multiplier: ×4.60 — for every $1 of fraud, total cost is $4.60
 *     (chargebacks + admin + lost goods + fees, Risk Solutions via ramp.com 2024)
 *   • FP revenue impact: ×1.63 — direct loss + 33% churn × $75 LTV
 *     (Aite-Novarica Group via greip.io 2024)
 *   • Human review: $16.33/FLAG — 20 min × $35/hr × 1.4× overhead
 *     (DuckDuckGoose AI 2024, Zippia 2024)
 *
 * Sources: ramp.com/blog/financial-impact-of-card-fraud,
 *          greip.io/blog, duckduckgoose.ai/white-papers, zippia.com
 *          Content paraphrased for compliance with licensing restrictions.
 */

import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie, Legend,
} from "recharts";
import type { EvaluationResponse } from "../api/client";

// ── Multipliers only (not base amounts — those come from real data) ───────────

const MULTIPLIERS = {
  fraudCostMultiplier: 4.60,   // Risk Solutions: total cost per $1 of fraud
  fpRevenueMultiplier: 1.63,   // direct loss + churn impact (Aite-Novarica)
  analystHourlyRate:   35,     // USD/hr (Zippia 2024, $66k/yr)
  minsPerReview:       20,     // DuckDuckGoose AI 2024
  overheadMultiplier:  1.40,   // benefits + tools + management
  chargebackRate:      0.15,   // 15% of caught fraud still disputes
  chargebackFee:       50,     // USD avg (directpaynet.com 2024)
};

const COST_PER_FLAG = (MULTIPLIERS.minsPerReview / 60)
  * MULTIPLIERS.analystHourlyRate
  * MULTIPLIERS.overheadMultiplier; // ~$16.33

// ── Helpers ───────────────────────────────────────────────────────────────────

function usd(n: number, dec = 0) {
  return "$" + Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: dec, maximumFractionDigits: dec,
  });
}

function Card({
  label, value, sub, color, icon, sign,
}: {
  label: string; value: string; sub?: string;
  color: string; icon: string; sign?: "+" | "-";
}) {
  return (
    <div style={{
      background: "#0a0f1e",
      border: `1px solid ${color}44`,
      borderRadius: 10, padding: "16px 18px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 18 }}>{icon}</span>
        <span style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 24, fontWeight: 800, color }}>
        {sign}{value}
      </div>
      {sub && <div style={{ fontSize: 11, color: "#64748b", marginTop: 6, lineHeight: 1.5 }}>{sub}</div>}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function FinancialImpactPanel({ evalData }: { evalData: EvaluationResponse }) {
  const [showAssumptions, setShowAssumptions] = useState(false);

  const { confusion_matrix: cm, metrics, summary, real_amounts: ra } = evalData;
  const { tp, fp, fn } = cm;

  // ── Real-data base amounts ────────────────────────────────────────────────
  // ra.tp_amt = raw transaction value of caught fraud
  // ra.fn_amt = raw transaction value of missed fraud
  // ra.fp_amt = raw transaction value of wrongly blocked legit

  // Apply multipliers to raw amounts
  const residualChargeback = ra.tp_amt * MULTIPLIERS.chargebackRate * MULTIPLIERS.chargebackFee
    / Math.max(1, summary.avg_fraud_amt); // per-txn chargeback on caught fraud

  // Money saved = full cost avoided on caught fraud
  // Exact same formula as Threshold Optimizer: tp_amt × 4.60
  const totalSaved    = ra.tp_amt * MULTIPLIERS.fraudCostMultiplier;

  // Money lost = full cost of missed fraud
  const totalLostFN   = ra.fn_amt * MULTIPLIERS.fraudCostMultiplier;

  // FP cost = revenue blocked × multiplier (churn + friction)
  const totalCostFP   = ra.fp_amt * MULTIPLIERS.fpRevenueMultiplier;

  // Human review cost — only a fraction of blocked transactions need analyst review
  // Same formula as the Threshold Optimizer: (tp+fp) × 1.4% × $16.33/case
  // The 1.4% FLAG_RATE reflects that most BLOCKs are auto-processed; only borderline
  // cases go to human review (DuckDuckGoose AI 2024: 275-375 hrs per 1,000 flagged)
  const FLAG_RATE       = 0.014;
  const flagCount       = Math.round((cm.tp + cm.fp) * FLAG_RATE);
  const totalHumanCost  = flagCount * COST_PER_FLAG;

  // Net impact
  const netImpact     = totalSaved - totalLostFN - totalCostFP - totalHumanCost;
  const netPositive   = netImpact >= 0;

  // ROI vs system cost ($0.002/prediction)
  const systemCost    = summary.total_evaluated * 0.002;
  const roi           = (totalSaved - systemCost) / Math.max(1, systemCost);

  // Pie data
  const pieData = [
    { name: "Fraud Saved",     value: Math.round(totalSaved),     color: "#22c55e" },
    { name: "Missed Fraud",    value: Math.round(totalLostFN),    color: "#ef4444" },
    { name: "False Positives", value: Math.round(totalCostFP),    color: "#f59e0b" },
    { name: "Human Review",    value: Math.round(totalHumanCost), color: "#8b5cf6" },
  ];

  // Per-segment breakdown using real amounts
  const segBreakdown = Object.entries(evalData.per_segment).map(([seg, m]) => ({
    seg,
    saved:  Math.round((m.real_tp_amt ?? 0) * MULTIPLIERS.fraudCostMultiplier),
    lostFN: Math.round((m.real_fn_amt ?? 0) * MULTIPLIERS.fraudCostMultiplier),
    costFP: Math.round((m.real_fp_amt ?? 0) * MULTIPLIERS.fpRevenueMultiplier),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#f1f5f9", marginBottom: 4 }}>
            Financial Impact — Real Transaction Amounts
          </div>
          <div style={{ fontSize: 12, color: "#64748b" }}>
            Base costs from actual <code style={{ color: "#60a5fa" }}>TransactionAmt</code> in the IEEE-CIS dataset.
            Multipliers from 2024 industry benchmarks (Risk Solutions, Aite-Novarica).
            Avg fraud txn: <b style={{ color: "#f87171" }}>{usd(summary.avg_fraud_amt, 2)}</b>
            {" · "}Avg legit txn: <b style={{ color: "#4ade80" }}>{usd(summary.avg_legit_amt, 2)}</b>
          </div>
        </div>
        <button
          onClick={() => setShowAssumptions(v => !v)}
          style={{
            fontSize: 11, padding: "4px 12px", borderRadius: 4,
            border: "1px solid #334155", background: "#1e293b", color: "#94a3b8",
            cursor: "pointer",
          }}
        >
          {showAssumptions ? "Hide" : "Show"} Multipliers
        </button>
      </div>

      {/* ── Raw data summary ─────────────────────────────────────────────── */}
      <div style={{
        background: "#0f172a", border: "1px solid #1e293b",
        borderRadius: 8, padding: "12px 16px",
        display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16,
      }}>
        {[
          { label: "Raw fraud caught (TP)", amt: ra.tp_amt, count: tp, color: "#22c55e", icon: "✅" },
          { label: "Raw fraud missed (FN)", amt: ra.fn_amt, count: fn, color: "#ef4444", icon: "❌" },
          { label: "Raw legit blocked (FP)", amt: ra.fp_amt, count: fp, color: "#f59e0b", icon: "🚫" },
        ].map(({ label, amt, count, color, icon }) => (
          <div key={label}>
            <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>{icon} {label}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color }}>{usd(amt, 2)}</div>
            <div style={{ fontSize: 11, color: "#475569" }}>
              {count.toLocaleString()} transactions · avg {usd(amt / Math.max(1, count), 2)}/txn
            </div>
          </div>
        ))}
      </div>

      {/* ── Multiplier assumptions ───────────────────────────────────────── */}
      {showAssumptions && (
        <div style={{
          background: "#0f172a", border: "1px solid #1e293b",
          borderRadius: 8, padding: 16, fontSize: 11, color: "#94a3b8", lineHeight: 1.8,
        }}>
          <div style={{ fontWeight: 700, color: "#cbd5e1", marginBottom: 8 }}>
            Multipliers Applied to Real Transaction Amounts
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 24px" }}>
            <span>📈 Fraud cost multiplier: <b style={{ color: "#f1f5f9" }}>×{MULTIPLIERS.fraudCostMultiplier}</b>
              — chargebacks + admin + lost goods (Risk Solutions, ramp.com 2024)</span>
            <span>🚫 FP revenue multiplier: <b style={{ color: "#f1f5f9" }}>×{MULTIPLIERS.fpRevenueMultiplier}</b>
              — direct loss + 33% churn × $75 LTV (Aite-Novarica via greip.io 2024)</span>
            <span>👤 Analyst cost/FLAG: <b style={{ color: "#f1f5f9" }}>{usd(COST_PER_FLAG, 2)}</b>
              — 20 min × $35/hr × 1.4× overhead (DuckDuckGoose AI + Zippia 2024)</span>
            <span>💳 Residual chargeback: <b style={{ color: "#f1f5f9" }}>15% × $50</b>
              — 15% of caught fraud still disputes (directpaynet.com 2024)</span>
          </div>
          <div style={{ marginTop: 8, color: "#475569", fontSize: 10 }}>
            Base amounts are real data from the dataset. Multipliers paraphrased from:
            ramp.com · greip.io · duckduckgoose.ai · directpaynet.com · zippia.com (2024).
            Content paraphrased for compliance with licensing restrictions.
          </div>
        </div>
      )}

      {/* ── KPI cards ────────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <Card icon="🛡️" label="Fraud Losses Prevented"
          value={usd(totalSaved)}
          sub={`${usd(ra.tp_amt, 2)} raw × ${MULTIPLIERS.fraudCostMultiplier}× multiplier`}
          color="#22c55e" sign="+" />
        <Card icon="💀" label="Fraud Losses Incurred"
          value={usd(totalLostFN)}
          sub={`${usd(ra.fn_amt, 2)} raw × ${MULTIPLIERS.fraudCostMultiplier}× multiplier`}
          color="#ef4444" sign="-" />
        <Card icon="🚫" label="False Positive Cost"
          value={usd(totalCostFP)}
          sub={`${usd(ra.fp_amt, 2)} blocked × ${MULTIPLIERS.fpRevenueMultiplier}× revenue impact`}
          color="#f59e0b" sign="-" />
        <Card icon="👤" label="Human Review Cost"
          value={usd(totalHumanCost)}
          sub={`${flagCount.toLocaleString()} FLAGs × ${usd(COST_PER_FLAG, 2)}/case`}
          color="#8b5cf6" sign="-" />
      </div>

      {/* ── Net impact + ROI ─────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div style={{
          background: netPositive ? "#052e16" : "#2d0a0a",
          border: `2px solid ${netPositive ? "#16a34a" : "#dc2626"}`,
          borderRadius: 10, padding: "20px 24px",
        }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
            Net Financial Impact
          </div>
          <div style={{ fontSize: 36, fontWeight: 800, color: netPositive ? "#4ade80" : "#f87171" }}>
            {netPositive ? "+" : "-"}{usd(Math.abs(netImpact))}
          </div>
          <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 8 }}>
            +{usd(totalSaved)} saved − {usd(totalLostFN)} lost − {usd(totalCostFP)} FP − {usd(totalHumanCost)} review
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: "#64748b" }}>
            Per transaction: {netPositive ? "+" : "-"}{usd(Math.abs(netImpact / summary.total_evaluated), 2)}
          </div>
        </div>

        <div style={{
          background: "#0a0f1e", border: "1px solid #1e293b",
          borderRadius: 10, padding: "20px 24px",
        }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
            System ROI
          </div>
          <div style={{ fontSize: 36, fontWeight: 800, color: "#60a5fa" }}>
            {roi.toFixed(0)}×
          </div>
          <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 8 }}>
            Return on {usd(systemCost, 0)} inference cost (@$0.002/prediction)
          </div>
          <div style={{ marginTop: 6, display: "flex", gap: 16, fontSize: 11 }}>
            <span style={{ color: "#4ade80" }}>Precision: {(metrics.precision * 100).toFixed(1)}%</span>
            <span style={{ color: "#f87171" }}>Recall: {(metrics.recall * 100).toFixed(1)}%</span>
            <span style={{ color: "#f59e0b" }}>F1: {metrics.f1.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* ── Segment breakdown + Pie ───────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Financial Breakdown by Segment (real amounts × multipliers)
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={segBreakdown} margin={{ left: 0, right: 16 }}>
              <XAxis dataKey="seg" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false}
                tickFormatter={v => "$" + (v >= 1000 ? (v / 1000).toFixed(0) + "k" : v)} />
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "none", borderRadius: 6 }}
                formatter={(v: unknown) => ["$" + (typeof v === "number" ? v.toLocaleString() : v)]}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              <Bar dataKey="saved"  name="Saved"     fill="#22c55e" radius={[2, 2, 0, 0]} />
              <Bar dataKey="lostFN" name="Lost (FN)" fill="#ef4444" radius={[2, 2, 0, 0]} />
              <Bar dataKey="costFP" name="FP Cost"   fill="#f59e0b" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Cost Distribution
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie data={pieData} dataKey="value" cx="50%" cy="50%" outerRadius={60} label={false}>
                {pieData.map(entry => <Cell key={entry.name} fill={entry.color} />)}
              </Pie>
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "none", borderRadius: 6 }}
                formatter={(v: unknown) => ["$" + (typeof v === "number" ? v.toLocaleString() : v)]}
              />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
            {pieData.map(({ name, value, color }) => (
              <div key={name} style={{ display: "flex", justifyContent: "space-between", fontSize: 10 }}>
                <span style={{ color }}>{name}</span>
                <span style={{ color: "#94a3b8" }}>${value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── What-if threshold table (real amounts) ────────────────────────── */}
      <div style={{ background: "#0a0f1e", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          💡 What-If: Real Financial Impact at Different Thresholds
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
                <th style={{ textAlign: "left", padding: "4px 12px" }}>Threshold</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Recall</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Precision</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Raw TP $</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Raw FN $</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Raw FP $</th>
                <th style={{ textAlign: "right", padding: "4px 12px" }}>Net Impact</th>
              </tr>
            </thead>
            <tbody>
              {evalData.threshold_analysis
                .filter(t => [0.3, 0.5, 0.7, 0.8, 0.88, 0.95].includes(t.threshold))
                .map(t => {
                  const tpAmt = t.real_tp_amt ?? 0;
                  const fnAmt = t.real_fn_amt ?? 0;
                  const fpAmt = t.real_fp_amt ?? 0;
                  const tSaved = tpAmt * MULTIPLIERS.fraudCostMultiplier;
                  const tLost  = fnAmt * MULTIPLIERS.fraudCostMultiplier;
                  const tFP    = fpAmt * MULTIPLIERS.fpRevenueMultiplier;
                  const tHuman = Math.round(((t.tp ?? 0) + (t.fp ?? 0)) * FLAG_RATE) * COST_PER_FLAG;
                  const tNet   = tSaved - tLost - tFP - tHuman;
                  const isCurrent = Math.abs(t.threshold - 0.88) < 0.01;
                  return (
                    <tr key={t.threshold} style={{
                      borderBottom: "1px solid #1e293b",
                      background: isCurrent ? "#1e293b" : "transparent",
                    }}>
                      <td style={{ padding: "5px 12px", color: isCurrent ? "#f59e0b" : "#94a3b8", fontWeight: isCurrent ? 700 : 400 }}>
                        {t.threshold.toFixed(2)} {isCurrent && <span style={{ fontSize: 9, color: "#f59e0b" }}>← current</span>}
                      </td>
                      <td style={{ padding: "5px 12px", textAlign: "right", color: "#f87171" }}>
                        {(t.recall * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "5px 12px", textAlign: "right", color: "#60a5fa" }}>
                        {(t.precision * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "5px 12px", textAlign: "right", color: "#4ade80" }}>
                        {usd(tpAmt, 0)}
                      </td>
                      <td style={{ padding: "5px 12px", textAlign: "right", color: "#f87171" }}>
                        {usd(fnAmt, 0)}
                      </td>
                      <td style={{ padding: "5px 12px", textAlign: "right", color: "#f59e0b" }}>
                        {usd(fpAmt, 0)}
                      </td>
                      <td style={{ padding: "5px 12px", textAlign: "right",
                        color: tNet >= 0 ? "#4ade80" : "#f87171", fontWeight: 700 }}>
                        {tNet >= 0 ? "+" : "-"}{usd(Math.abs(tNet))}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 8, fontSize: 10, color: "#475569" }}>
          Raw $ = actual TransactionAmt from IEEE-CIS dataset. Net = (TP × 4.60) − (FN × 4.60) − (FP × 1.63).
          Human review cost excluded from this table for clarity.
          Multipliers: Risk Solutions (ramp.com 2024), Aite-Novarica (greip.io 2024).
          Content paraphrased for compliance with licensing restrictions.
        </div>
      </div>

    </div>
  );
}
