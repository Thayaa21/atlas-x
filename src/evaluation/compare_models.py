"""
ATLAS-X – Model Comparison: v3 vs v4 (graph-augmented)
────────────────────────────────────────────────────────
Tests both models on the same 20% stratified holdout.

  v3 (baseline)  : src/models/atlass_x_xgb_v3.pkl
                   evaluated on train_full_features.parquet
  v4 (graph)     : src/models/atlass_x_xgb_v4_graph.pkl
                   evaluated on train_with_graph_features.parquet

Both splits use random_state=42 and stratify=y so the holdout rows are
identical — the only difference is the feature matrix.

Output:
  results/model_comparison.json

Usage:
    python -m src.evaluation.compare_models
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.optimization.threshold_optimizer import compute_customer_segments

REPO     = Path(__file__).resolve().parents[2]
OUT_DIR  = REPO / "results"

V3_MODEL  = REPO / "src/models/atlass_x_xgb_v3.pkl"
V4_MODEL  = REPO / "src/models/atlass_x_xgb_v4_graph.pkl"
V3_DATA   = REPO / "data/processed/train_full_features.parquet"
V4_DATA   = REPO / "data/processed/train_with_graph_features.parquet"
THRESH_P  = REPO / "src/optimization/artifacts/thresholds.json"

GRAPH_COLS = [
    "device_fraud_rate", "device_card_velocity", "connected_fraud_cards",
    "email_fraud_rate", "address_fraud_rate", "graph_risk_score",
]

DROP_COLS = ["isFraud", "TransactionID", "TransactionDT", "customer_segment"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame):
    """Return X, y, seg_test for the 20% holdout."""
    segment_s, _ = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = segment_s

    X = df.drop([c for c in DROP_COLS if c in df.columns], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    _, X_test, _, y_test, _, seg_test = train_test_split(
        X, y, df["customer_segment"],
        test_size=0.2, random_state=42, stratify=y,
    )
    return X_test, y_test, seg_test.reset_index(drop=True)


def _predict(model, X_test, seg_test, thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = np.zeros(len(y_prob), dtype=int)
    for seg in ["VIP", "Regular", "New"]:
        mask = (seg_test == seg).values
        y_pred[mask] = (y_prob[mask] >= thresholds[seg]).astype(int)
    return y_prob, y_pred


def _metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ── Bar-chart comparison ──────────────────────────────────────────────────────

def _plot_comparison(v3: dict, v4: dict, path: Path) -> None:
    metrics_to_show = ["roc_auc", "precision", "recall", "f1"]
    labels          = ["AUC", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_v3 = ax.bar(x - width/2, [v3[m] for m in metrics_to_show], width,
                     label="v3 (baseline)", color="#4C72B0", alpha=0.85)
    bars_v4 = ax.bar(x + width/2, [v4[m] for m in metrics_to_show], width,
                     label="v4 (graph features)", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ATLAS-X  v3 vs v4  –  20% Holdout")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars_v3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_v4:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Feature importance delta ──────────────────────────────────────────────────

def _graph_feature_importance(model_v4, feature_names: list) -> dict:
    """Return importance scores for only the 6 graph features."""
    imp = dict(zip(feature_names, model_v4.feature_importances_))
    return {col: round(float(imp.get(col, 0.0)), 6) for col in GRAPH_COLS}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not V4_MODEL.exists():
        raise FileNotFoundError(
            f"{V4_MODEL} not found.\n"
            "Run  python -m src.models.train_v4_with_graph  first."
        )

    thresh_art = json.loads(THRESH_P.read_text())
    thresholds = {s: float(thresh_art["segments"][s]["threshold"]) for s in ["VIP", "Regular", "New"]}

    # ── v3 ────────────────────────────────────────────────────────────────────
    print("Evaluating v3 (baseline)...")
    model_v3 = joblib.load(V3_MODEL)
    df_v3    = pd.read_parquet(V3_DATA)
    X_v3, y_v3, seg_v3 = _prep(df_v3)
    y_prob_v3, y_pred_v3 = _predict(model_v3, X_v3, seg_v3, thresholds)
    m_v3 = _metrics(y_v3, y_pred_v3, y_prob_v3)
    print(f"  Recall={m_v3['recall']:.2%}  Precision={m_v3['precision']:.2%}  F1={m_v3['f1']:.2%}  AUC={m_v3['roc_auc']:.4f}")

    # ── v4 ────────────────────────────────────────────────────────────────────
    print("Evaluating v4 (graph features)...")
    model_v4 = joblib.load(V4_MODEL)
    df_v4    = pd.read_parquet(V4_DATA)
    # Graph feature columns are numeric — fill any NaN before predict
    for col in GRAPH_COLS:
        if col in df_v4.columns:
            df_v4[col] = df_v4[col].fillna(0.0)
    X_v4, y_v4, seg_v4 = _prep(df_v4)
    y_prob_v4, y_pred_v4 = _predict(model_v4, X_v4, seg_v4, thresholds)
    m_v4 = _metrics(y_v4, y_pred_v4, y_prob_v4)
    print(f"  Recall={m_v4['recall']:.2%}  Precision={m_v4['precision']:.2%}  F1={m_v4['f1']:.2%}  AUC={m_v4['roc_auc']:.4f}")

    # ── Delta ─────────────────────────────────────────────────────────────────
    delta = {
        "recall":    round(m_v4["recall"]    - m_v3["recall"],    4),
        "precision": round(m_v4["precision"] - m_v3["precision"], 4),
        "f1":        round(m_v4["f1"]        - m_v3["f1"],        4),
        "roc_auc":   round(m_v4["roc_auc"]   - m_v3["roc_auc"],   4),
        "delta_tp":  m_v4["tp"] - m_v3["tp"],
        "delta_fp":  m_v4["fp"] - m_v3["fp"],
        "delta_fn":  m_v4["fn"] - m_v3["fn"],
    }

    # ── Graph feature importance ──────────────────────────────────────────────
    feature_names = list(X_v4.columns)
    gf_imp = _graph_feature_importance(model_v4, feature_names)
    total_features = len(feature_names)
    print(f"\nGraph feature importances (out of {total_features} total features):")
    for col, imp in sorted(gf_imp.items(), key=lambda x: -x[1]):
        print(f"  {col:<32} {imp:.6f}")

    # ── Console table ─────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  MODEL COMPARISON  –  20% holdout")
    print("═" * 60)
    print(f"  {'Metric':<22} {'v3 (baseline)':>14} {'v4 (graph)':>12} {'Δ':>8}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*12}  {'-'*8}")
    for key, label in [("roc_auc","AUC"), ("recall","Recall"), ("precision","Precision"), ("f1","F1")]:
        print(f"  {label:<22} {m_v3[key]:>14.2%} {m_v4[key]:>12.2%} {delta[key]:>+8.2%}")
    print(f"  {'False Negatives':<22} {m_v3['fn']:>14,} {m_v4['fn']:>12,} {delta['delta_fn']:>+8,}")
    print(f"  {'False Positives':<22} {m_v3['fp']:>14,} {m_v4['fp']:>12,} {delta['delta_fp']:>+8,}")
    print(f"  {'True Positives':<22} {m_v3['tp']:>14,} {m_v4['tp']:>12,} {delta['delta_tp']:>+8,}")
    print("═" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "holdout_size":      int(len(y_v3)),
        "total_fraud":       int(y_v3.sum()),
        "segment_thresholds": thresholds,
        "v3_baseline":       m_v3,
        "v4_graph":          m_v4,
        "delta":             delta,
        "graph_feature_importances": gf_imp,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "model_comparison.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → results/model_comparison.json")

    _plot_comparison(m_v3, m_v4, OUT_DIR / "model_comparison.png")
    print(f"Saved → results/model_comparison.png")


if __name__ == "__main__":
    main()
