"""
ATLAS-X Model Validation
Reproduces the exact 20% holdout split used during training and measures
all baseline metrics. Outputs JSON + 3 visualisation PNGs to results/.
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
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.optimization.threshold_optimizer import compute_customer_segments

REPO = Path(__file__).resolve().parent
DATA_PATH = REPO / "data/processed/train_full_features.parquet"
MODEL_PATH = REPO / "src/models/atlass_x_xgb_v3.pkl"
THRESH_PATH = REPO / "src/optimization/artifacts/thresholds.json"
OUT_DIR = REPO / "results"

PRECISION_MIN = 0.70
RECALL_MIN = 0.75


def load_holdout():
    df = pd.read_parquet(DATA_PATH)

    segment_s, _ = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = segment_s

    X = df.drop(["isFraud", "TransactionID", "TransactionDT", "customer_segment"], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    _, X_test, _, y_test, _, seg_test = train_test_split(
        X, y, df["customer_segment"],
        test_size=0.2, random_state=42, stratify=y,
    )
    return X_test, y_test, seg_test


def apply_segment_thresholds(y_prob, seg_test, thresholds):
    pred = np.zeros(len(y_prob), dtype=int)
    for seg in ["VIP", "Regular", "New"]:
        mask = (seg_test == seg).values
        pred[mask] = (y_prob[mask] >= thresholds[seg]).astype(int)
    return pred


def metrics_dict(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fdr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_decline_rate": fdr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_t = None
    best_f1 = -1.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= PRECISION_MIN and r >= RECALL_MIN:
            f1 = 2 * p * r / (p + r + 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

    # fallback: best F1 with penalty if constraints can't be met simultaneously
    if best_t is None:
        best_score = -np.inf
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
            f1 = 2 * p * r / (p + r + 1e-12)
            penalty = max(0.0, PRECISION_MIN - p) + max(0.0, RECALL_MIN - r)
            score = f1 - 3.0 * penalty
            if score > best_score:
                best_score = score
                best_t = t

    return float(best_t)


def plot_confusion_matrix(cm_dict, path):
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["Legit", "Fraud"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Optimal Threshold)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost v3")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(y_true, y_prob, optimal_t, path):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="#16A34A", lw=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.axhline(PRECISION_MIN, color="gray", ls="--", lw=1, label=f"Precision ≥ {PRECISION_MIN}")
    ax.axvline(RECALL_MIN, color="gray", ls=":",  lw=1, label=f"Recall ≥ {RECALL_MIN}")

    # Mark the optimal threshold point
    idx = np.searchsorted(thresholds, optimal_t)
    if idx < len(recalls) - 1:
        ax.scatter(recalls[idx], precisions[idx], color="red", zorder=5,
                   label=f"Optimal t={optimal_t:.3f}")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — XGBoost v3")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading holdout split...")
    X_test, y_test, seg_test = load_holdout()

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Segment threshold metrics ---
    thresholds_art = json.loads(THRESH_PATH.read_text())
    seg_thresholds = {s: float(thresholds_art["segments"][s]["threshold"]) for s in ["VIP", "Regular", "New"]}
    y_pred_seg = apply_segment_thresholds(y_prob, seg_test, seg_thresholds)
    seg_metrics = metrics_dict(y_test, y_pred_seg, y_prob)

    # --- Optimal single threshold from PR curve ---
    print("Finding optimal threshold...")
    optimal_t = find_optimal_threshold(y_test, y_prob)
    y_pred_opt = (y_prob >= optimal_t).astype(int)
    opt_metrics = metrics_dict(y_test, y_pred_opt, y_prob)

    constraints_met = (
        opt_metrics["precision"] >= PRECISION_MIN and
        opt_metrics["recall"] >= RECALL_MIN
    )

    report = {
        "holdout_size": int(len(y_test)),
        "fraud_rate_in_holdout": float(y_test.mean()),
        "threshold_constraints": {
            "precision_min": PRECISION_MIN,
            "recall_min": RECALL_MIN,
            "constraints_met": constraints_met,
        },
        "segment_thresholds_used": seg_thresholds,
        "metrics_at_segment_thresholds": seg_metrics,
        "optimal_threshold": optimal_t,
        "metrics_at_optimal_threshold": opt_metrics,
    }

    out_json = OUT_DIR / "baseline_metrics.json"
    out_json.write_text(json.dumps(report, indent=2))

    # --- Plots ---
    print("Generating plots...")
    plot_confusion_matrix(opt_metrics["confusion_matrix"], OUT_DIR / "confusion_matrix.png")
    plot_roc(y_test, y_prob, OUT_DIR / "roc_curve.png")
    plot_pr(y_test, y_prob, optimal_t, OUT_DIR / "precision_recall_curve.png")

    # --- Console summary ---
    print("\n" + "=" * 50)
    print("SEGMENT THRESHOLDS")
    print(f"  AUC-ROC:     {seg_metrics['roc_auc']:.4f}")
    print(f"  Precision:   {seg_metrics['precision']:.4f}")
    print(f"  Recall:      {seg_metrics['recall']:.4f}")
    print(f"  F1:          {seg_metrics['f1']:.4f}")
    print(f"  FDR:         {seg_metrics['false_decline_rate']:.4f}")
    cm = seg_metrics["confusion_matrix"]
    print(f"  TN={cm['tn']:,}  FP={cm['fp']:,}  FN={cm['fn']:,}  TP={cm['tp']:,}")

    print(f"\nOPTIMAL THRESHOLD  (t={optimal_t:.4f})  constraints_met={constraints_met}")
    print(f"  AUC-ROC:     {opt_metrics['roc_auc']:.4f}")
    print(f"  Precision:   {opt_metrics['precision']:.4f}")
    print(f"  Recall:      {opt_metrics['recall']:.4f}")
    print(f"  F1:          {opt_metrics['f1']:.4f}")
    print(f"  FDR:         {opt_metrics['false_decline_rate']:.4f}")
    cm = opt_metrics["confusion_matrix"]
    print(f"  TN={cm['tn']:,}  FP={cm['fp']:,}  FN={cm['fn']:,}  TP={cm['tp']:,}")
    print("=" * 50)
    print(f"\nSaved → {out_json}")
    print(f"Saved → {OUT_DIR / 'confusion_matrix.png'}")
    print(f"Saved → {OUT_DIR / 'roc_curve.png'}")
    print(f"Saved → {OUT_DIR / 'precision_recall_curve.png'}")


if __name__ == "__main__":
    main()
