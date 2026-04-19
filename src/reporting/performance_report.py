import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.rl.dqn_agent import QNetwork, encode_segment, ACTIONS


repo_root = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def segment_from_cuts_vec(*, d1: np.ndarray, d15: np.ndarray, seg_params: Dict[str, float]) -> np.ndarray:
    eps = 1e-12
    d1_min = float(seg_params["d1_min"])
    d1_max = float(seg_params["d1_max"])
    d15_min = float(seg_params["d15_min"])
    d15_max = float(seg_params["d15_max"])
    vip_cut = float(seg_params["vip_cut"])
    regular_cut = float(seg_params["regular_cut"])

    tenure_score = (d1 - d1_min) / (d1_max - d1_min + eps)
    recency_score = (d15_max - d15) / (d15_max - d15_min + eps)
    combined = 0.5 * tenure_score + 0.5 * recency_score

    seg = np.where(combined >= vip_cut, "VIP", np.where(combined >= regular_cut, "Regular", "New"))
    return seg


def apply_segment_thresholds(*, y_true: np.ndarray, y_prob: np.ndarray, segments: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, Any]:
    pred = np.zeros_like(y_prob, dtype=int)
    for seg in ["VIP", "Regular", "New"]:
        mask = segments == seg
        t = float(thresholds[seg])
        pred[mask] = (y_prob[mask] >= t).astype(int)

    overall = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }

    per_segment = {}
    for seg in ["VIP", "Regular", "New"]:
        mask = segments == seg
        if int(mask.sum()) == 0:
            per_segment[seg] = {"error": "no samples"}
            continue
        y_t = y_true[mask]
        y_p = y_prob[mask]
        pred_s = pred[mask]
        per_segment[seg] = {
            "roc_auc": float(roc_auc_score(y_t, y_p)) if y_t.min() != y_t.max() else None,
            "precision": float(precision_score(y_t, pred_s, zero_division=0)),
            "recall": float(recall_score(y_t, pred_s, zero_division=0)),
            "f1": float(f1_score(y_t, pred_s, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_t, pred_s).tolist(),
        }
    return {"overall": overall, "per_segment": per_segment, "thresholds_used": thresholds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--output", type=str, default="reports/perf_report.json")
    args = parser.parse_args()

    sample_size = int(args.sample_size)
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_features_p = repo_root / "data/processed/train_full_features.parquet"
    thresholds_p = repo_root / "src/optimization/artifacts/thresholds.json"
    dqn_p = repo_root / "src/rl/trained_dqn.pth"
    model_p = repo_root / "src/models/atlass_x_xgb_v3.pkl"

    if not full_features_p.exists():
        raise FileNotFoundError(f"Missing dataset: {full_features_p}")
    if not thresholds_p.exists():
        raise FileNotFoundError(f"Missing thresholds artifact: {thresholds_p}")
    if not model_p.exists():
        raise FileNotFoundError(f"Missing model: {model_p}")
    if not dqn_p.exists():
        raise FileNotFoundError(f"Missing DQN: {dqn_p}")

    df = pd.read_parquet(full_features_p)
    if sample_size > 0 and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    thresholds_art = _load_json(thresholds_p)
    seg_params = thresholds_art["segment_quantile_cuts"]
    thresholds = {seg: float(thresholds_art["segments"][seg]["threshold"]) for seg in ["VIP", "Regular", "New"]}

    # Compute segments using persisted normalization stats
    segments = segment_from_cuts_vec(d1=df["D1"].to_numpy().astype(float), d15=df["D15"].to_numpy().astype(float), seg_params=seg_params)

    X = df.drop(["isFraud", "TransactionID", "TransactionDT"], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    # Cast categoricals as in model training
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    # Split for evaluation
    X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
        X, y, segments, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(model_p)

    # Vectorized prediction for metrics
    t0 = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]
    t_pred = time.time() - t0

    metrics = apply_segment_thresholds(
        y_true=y_test,
        y_prob=y_prob,
        segments=seg_test,
        thresholds=thresholds,
    )

    # Latency measurement: per-row model.predict_proba for a small sample
    per_row_n = min(200, len(X_test))
    X_small = X_test.iloc[:per_row_n].copy()
    lat = []
    for i in range(len(X_small)):
        row_df = X_small.iloc[[i]]
        t_start = time.time()
        _ = model.predict_proba(row_df)[:, 1][0]
        lat.append(time.time() - t_start)
    lat_ms = np.array(lat) * 1000.0
    latency_stats = {
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "avg_ms": float(np.mean(lat_ms)),
        "throughput_vec_txn_per_sec": float(len(X_test) / max(1e-9, t_pred)),
    }

    # DQN decision latency on the same sample (state vector computation + argmax)
    dqn_ckpt = torch.load(dqn_p, map_location="cpu")
    qnet = QNetwork(int(dqn_ckpt["input_dim"]), int(dqn_ckpt["output_dim"]))
    qnet.load_state_dict(dqn_ckpt["state_dict"])
    qnet.eval()

    # Build state vectors from available columns in X_test. graph_risk uses cluster_fraud_rate.
    dqn_lat = []
    action_counts = {a: 0 for a in ACTIONS}
    for i in range(per_row_n):
        t_start = time.time()
        row = X_small.iloc[i]
        fraud_p = float(model.predict_proba(X_small.iloc[[i]])[:, 1][0])
        graph_risk = float(row["cluster_fraud_rate"]) if "cluster_fraud_rate" in X_small.columns else 0.0
        seg = seg_test[i] if i < len(seg_test) else "New"
        seg_code = encode_segment(seg)
        market_context = float(row["Transaction_Hour"] / 23.0) if "Transaction_Hour" in X_small.columns else 0.0
        state = torch.tensor([[fraud_p, graph_risk, seg_code, market_context]], dtype=torch.float32)
        with torch.no_grad():
            q = qnet(state)
            a_idx = int(torch.argmax(q, dim=1).item())
        action_counts[ACTIONS[a_idx]] += 1
        dqn_lat.append(time.time() - t_start)

    dqn_lat_ms = np.array(dqn_lat) * 1000.0
    dqn_latency_stats = {
        "p50_ms": float(np.percentile(dqn_lat_ms, 50)),
        "p95_ms": float(np.percentile(dqn_lat_ms, 95)),
        "avg_ms": float(np.mean(dqn_lat_ms)),
        "action_counts": action_counts,
    }

    report = {
        "sample_size": sample_size,
        "dataset_rows": int(len(df)),
        "metrics": metrics,
        "latency": latency_stats,
        "dqn_latency": dqn_latency_stats,
    }

    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["metrics"]["overall"], indent=2))
    print("Latency stats:", latency_stats)
    print("DQN latency stats:", dqn_latency_stats)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

