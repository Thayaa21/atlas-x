"""
ATLAS-X Graph Integration Evaluation
──────────────────────────────────────
Measures the real impact of adding the Neo4j graph override layer to
the XGBoost + segment-threshold pipeline on the 20% holdout.

Pipeline:
  1. XGBoost score + segment threshold  → FRAUD / CLEAR
  2. For every CLEAR: batch-query Neo4j  → FLAG if card in ring (≥2 connections)
  3. Final decision: FRAUD | FLAG | CLEAR
     FRAUD + FLAG = treated as "predicted fraud" for metric purposes

Outputs:
  results/graph_integration_results.json
  results/neo4j_performance.json
  results/graph_confusion_matrix.png   (updated confusion matrix)

Usage:
    python -m src.evaluation.test_graph_integration
"""
import json
import time
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

from src.graph.fraud_ring_checker import FraudRingChecker
from src.optimization.threshold_optimizer import compute_customer_segments

REPO     = Path(__file__).resolve().parents[2]
OUT_DIR  = REPO / "results"
MODEL_P  = REPO / "src/models/atlass_x_xgb_v3.pkl"
DATA_P   = REPO / "data/processed/train_full_features.parquet"
THRESH_P = REPO / "src/optimization/artifacts/thresholds.json"

# Baseline from validate_model.py (segment threshold run)
BASELINE = {
    "recall":    0.5805,
    "precision": 0.8432,
    "f1":        0.6876,
    "fn":        1734,
    "fp":        446,
    "tn":        113529,
    "tp":        2399,
    "roc_auc":   0.9528,
}
FRAUD_COST_SAVING = 120   # $ per additional TP
FP_COST           = 50    # $ per additional FP


# ── Load holdout ──────────────────────────────────────────────────────────────

def load_holdout():
    df = pd.read_parquet(DATA_P)

    segment_s, _ = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = segment_s
    df["card_id"] = (
        df["card1"].astype(str).str.strip() + "-" +
        df["card2"].fillna("NA").astype(str).str.strip() + "-" +
        df["card3"].fillna("NA").astype(str).str.strip()
    )

    X = df.drop(["isFraud", "TransactionID", "TransactionDT", "customer_segment", "card_id"], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    _, X_test, _, y_test, seg_train, seg_test, _, card_test = train_test_split(
        X, y, df["customer_segment"], df["card_id"],
        test_size=0.2, random_state=42, stratify=y,
    )
    return X_test, y_test, seg_test.reset_index(drop=True), card_test.reset_index(drop=True)


# ── Baseline predictions (XGBoost + segment thresholds) ──────────────────────

def baseline_predict(X_test, y_test, seg_test):
    model      = joblib.load(MODEL_P)
    thresh_art = json.loads(THRESH_P.read_text())
    thresholds = {s: float(thresh_art["segments"][s]["threshold"]) for s in ["VIP", "Regular", "New"]}

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = np.zeros(len(y_prob), dtype=int)
    for seg in ["VIP", "Regular", "New"]:
        mask = (seg_test == seg).values
        y_pred[mask] = (y_prob[mask] >= thresholds[seg]).astype(int)

    return y_prob, y_pred


# ── Apply graph override ──────────────────────────────────────────────────────

def apply_graph_override(
    y_pred_base: np.ndarray,
    y_prob: np.ndarray,
    card_ids: pd.Series,
    checker: FraudRingChecker,
    min_fraud_prob: float = 0.15,
):
    """
    For all CLEAR decisions, batch-query Neo4j for device rings.
    Override to FLAG only when BOTH conditions hold:
      1. Card is connected to ≥2 fraud cards via a shared device
      2. fraud_prob ≥ min_fraud_prob  (model already sees some suspicion)

    Pure graph override is too noisy on this dataset — a card shared across
    many transactions inflates FPs dramatically. Combining with a probability
    floor keeps the precision/recall tradeoff sensible.
    """
    clear_mask  = y_pred_base == 0
    clear_cards = card_ids.values[clear_mask]

    print(f"  Batch-querying {len(clear_cards):,} CLEAR cards ({len(set(clear_cards)):,} unique)...")
    print(f"  Mode: device-only connections, min_fraud_prob={min_fraud_prob}")
    t0 = time.perf_counter()
    flagged_map = checker.check_batch(clear_cards.tolist(), threshold=2, device_only=True)
    elapsed = time.perf_counter() - t0
    print(f"  Batch query took {elapsed:.2f}s — {len(flagged_map):,} unique device-ring cards")

    y_pred_enhanced = y_pred_base.copy()
    flagged_indices = []
    for i, (cid, is_clear) in enumerate(zip(card_ids.values, clear_mask)):
        if is_clear and cid in flagged_map and y_prob[i] >= min_fraud_prob:
            y_pred_enhanced[i] = 1
            flagged_indices.append(i)

    return y_pred_enhanced, flagged_map, flagged_indices


# ── Metrics helper ────────────────────────────────────────────────────────────

def calc_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_decline_rate": round(float(fp / (fp + tn)), 4),
    }


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_cm(base_m: dict, enh_m: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    titles = ["Baseline (XGBoost only)", "With Graph Layer"]
    for ax, m, title in zip(axes, [base_m, enh_m], titles):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        labels = ["Legit", "Fraud"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{title}\nRecall={m['recall']:.2%}  Prec={m['precision']:.2%}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Neo4j latency benchmark ───────────────────────────────────────────────────

def benchmark_latency(checker: FraudRingChecker, card_ids: pd.Series, n: int = 1000) -> dict:
    sample = card_ids.drop_duplicates().sample(min(n, card_ids.nunique()), random_state=42).tolist()
    checker._cached_check.cache_clear()   # force cold queries

    latencies = []
    for cid in sample:
        t0 = time.perf_counter()
        checker.check(cid)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    return {
        "n_queries":  len(lat),
        "avg_ms":     round(float(lat.mean()), 2),
        "p50_ms":     round(float(np.percentile(lat, 50)), 2),
        "p95_ms":     round(float(np.percentile(lat, 95)), 2),
        "p99_ms":     round(float(np.percentile(lat, 99)), 2),
        "max_ms":     round(float(lat.max()), 2),
        "target_met": bool(float(np.percentile(lat, 95)) < 50),
    }


# ── Console banner ────────────────────────────────────────────────────────────

def print_banner(base: dict, enh: dict, perf: dict) -> None:
    delta_tp  = enh["tp"]  - base["tp"]
    delta_fp  = enh["fp"]  - base["fp"]
    delta_fn  = enh["fn"]  - base["fn"]
    net_saving = delta_tp * FRAUD_COST_SAVING - delta_fp * FP_COST

    print()
    print("═" * 52)
    print("  GRAPH INTEGRATION TEST RESULTS")
    print("═" * 52)
    print(f"  {'METRIC':<22} {'BASELINE':>10} {'WITH GRAPH':>10} {'CHANGE':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'Recall':<22} {base['recall']:>10.2%} {enh['recall']:>10.2%} {enh['recall']-base['recall']:>+8.2%}")
    print(f"  {'Precision':<22} {base['precision']:>10.2%} {enh['precision']:>10.2%} {enh['precision']-base['precision']:>+8.2%}")
    print(f"  {'F1-Score':<22} {base['f1']:>10.2%} {enh['f1']:>10.2%} {enh['f1']-base['f1']:>+8.2%}")
    print(f"  {'False Negatives':<22} {base['fn']:>10,} {enh['fn']:>10,} {enh['fn']-base['fn']:>+8,}")
    print(f"  {'False Positives':<22} {base['fp']:>10,} {enh['fp']:>10,} {enh['fp']-base['fp']:>+8,}")
    print()
    print(f"  Additional frauds caught : +{delta_tp:,}")
    print(f"  Additional false positives: +{delta_fp:,}")
    print(f"  Net financial impact     : +{delta_tp}×${FRAUD_COST_SAVING} − {delta_fp}×${FP_COST} = ${net_saving:,}")
    print()
    print(f"  Neo4j p95 latency : {perf['p95_ms']}ms  (target <50ms: {'✓' if perf['target_met'] else '✗'})")
    print("═" * 52)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading holdout split...")
    X_test, y_test, seg_test, card_test = load_holdout()
    print(f"  {len(y_test):,} transactions, {int(y_test.sum()):,} fraud")

    print("Running baseline XGBoost predictions...")
    y_prob, y_pred_base = baseline_predict(X_test, y_test, seg_test)
    base_metrics = calc_metrics(y_test, y_pred_base, y_prob)
    print(f"  Recall={base_metrics['recall']:.2%}  Precision={base_metrics['precision']:.2%}  F1={base_metrics['f1']:.2%}")

    print("Connecting to Neo4j...")
    checker = FraudRingChecker.get()

    print("Applying graph override (device-only, prob≥0.15)...")
    y_pred_enh, flagged_map, flagged_idx = apply_graph_override(
        y_pred_base, y_prob, card_test, checker, min_fraud_prob=0.15
    )
    enh_metrics = calc_metrics(y_test, y_pred_enh, y_prob)
    print(f"  Recall={enh_metrics['recall']:.2%}  Precision={enh_metrics['precision']:.2%}  F1={enh_metrics['f1']:.2%}")

    print("Benchmarking Neo4j latency (1,000 queries)...")
    perf = benchmark_latency(checker, card_test)
    print(f"  avg={perf['avg_ms']}ms  p50={perf['p50_ms']}ms  p95={perf['p95_ms']}ms  p99={perf['p99_ms']}ms")

    # ── Delta analysis ────────────────────────────────────────────────────────
    delta_tp = enh_metrics["tp"] - base_metrics["tp"]
    delta_fp = enh_metrics["fp"] - base_metrics["fp"]
    net_saving = delta_tp * FRAUD_COST_SAVING - delta_fp * FP_COST

    result = {
        "holdout_size": int(len(y_test)),
        "total_fraud_in_holdout": int(y_test.sum()),
        "cards_flagged_by_graph": len(flagged_map),
        "transactions_overridden": len(flagged_idx),
        "baseline":  base_metrics,
        "with_graph": enh_metrics,
        "delta": {
            "recall":    round(enh_metrics["recall"]    - base_metrics["recall"],    4),
            "precision": round(enh_metrics["precision"] - base_metrics["precision"], 4),
            "f1":        round(enh_metrics["f1"]        - base_metrics["f1"],        4),
            "delta_tp":  delta_tp,
            "delta_fp":  delta_fp,
            "delta_fn":  enh_metrics["fn"] - base_metrics["fn"],
        },
        "financial_impact": {
            "additional_frauds_caught": delta_tp,
            "additional_false_positives": delta_fp,
            "fraud_savings_usd":   delta_tp * FRAUD_COST_SAVING,
            "fp_cost_usd":         delta_fp * FP_COST,
            "net_saving_usd":      net_saving,
        },
    }

    (OUT_DIR / "graph_integration_results.json").write_text(json.dumps(result, indent=2))
    (OUT_DIR / "neo4j_performance.json").write_text(json.dumps(perf, indent=2))

    plot_cm(base_metrics, enh_metrics, OUT_DIR / "graph_confusion_matrix.png")

    print_banner(base_metrics, enh_metrics, perf)

    print(f"Saved → results/graph_integration_results.json")
    print(f"Saved → results/neo4j_performance.json")
    print(f"Saved → results/graph_confusion_matrix.png")


if __name__ == "__main__":
    main()
