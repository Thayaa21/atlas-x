"""
ATLAS-X – XGBoost v4 (Graph-Augmented)
────────────────────────────────────────
Retrains XGBoost with the 6 graph topology features added by
src/features/engineer_graph_features.py.

All hyperparameters are identical to v3 so the comparison is fair.
The only difference is the input feature matrix.

Usage:
    python -m src.models.train_v4_with_graph
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.optimization.threshold_optimizer import compute_customer_segments

REPO      = Path(__file__).resolve().parents[2]
DATA_P    = REPO / "data/processed/train_with_graph_features.parquet"
MODEL_OUT = REPO / "src/models/atlass_x_xgb_v4_graph.pkl"
THRESH_P  = REPO / "src/optimization/artifacts/thresholds.json"
METRICS_P = REPO / "results/v4_graph_metrics.json"

GRAPH_COLS = [
    "device_fraud_rate",
    "device_card_velocity",
    "connected_fraud_cards",
    "email_fraud_rate",
    "address_fraud_rate",
    "graph_risk_score",
]


def load_and_split():
    df = pd.read_parquet(DATA_P)

    segment_s, _ = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = segment_s

    X = df.drop(["isFraud", "TransactionID", "TransactionDT", "customer_segment"], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    # Cast categorical / object columns (same treatment as v3)
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    # Graph feature columns are already float64 – ensure no NaN
    for col in GRAPH_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)

    X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
        X, y, df["customer_segment"],
        test_size=0.2, random_state=42, stratify=y,
    )
    return X_train, X_test, y_train, y_test, seg_train, seg_test.reset_index(drop=True)


def train(X_train, y_train, X_test, y_test):
    fraud_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    print(f"  scale_pos_weight = {fraud_weight:.2f}  "
          f"(train fraud rate = {y_train.mean():.3%})")

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=fraud_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    return model


def evaluate(model, X_test, y_test, seg_test) -> dict:
    # Load segment thresholds calibrated for v3 — same thresholds for fair comparison
    thresh_art = json.loads(THRESH_P.read_text())
    thresholds = {s: float(thresh_art["segments"][s]["threshold"]) for s in ["VIP", "Regular", "New"]}

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = np.zeros(len(y_prob), dtype=int)
    for seg in ["VIP", "Regular", "New"]:
        mask = (seg_test == seg).values
        y_pred[mask] = (y_prob[mask] >= thresholds[seg]).astype(int)

    cm           = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "thresholds_used": thresholds,
    }


def main():
    if not DATA_P.exists():
        raise FileNotFoundError(
            f"{DATA_P} not found.\n"
            "Run  python -m src.features.engineer_graph_features  first."
        )

    print(f"Loading {DATA_P.name}...")
    X_train, X_test, y_train, y_test, seg_train, seg_test = load_and_split()
    print(f"  Train: {len(y_train):,}  |  Test: {len(y_test):,}")

    # Show which graph columns are present
    graph_present = [c for c in GRAPH_COLS if c in X_train.columns]
    print(f"  Graph features added: {graph_present}")

    print("\nTraining XGBoost v4 (same hyperparams as v3)...")
    model = train(X_train, y_train, X_test, y_test)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    print("Evaluating on holdout (segment thresholds from v3 calibration)...")
    metrics = evaluate(model, X_test, y_test, seg_test)

    METRICS_P.parent.mkdir(parents=True, exist_ok=True)
    METRICS_P.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved → {METRICS_P}")

    print()
    print("═" * 42)
    print("  XGBoost v4 (graph features) results")
    print("═" * 42)
    print(f"  AUC       : {metrics['roc_auc']:.4f}")
    print(f"  Recall    : {metrics['recall']:.2%}")
    print(f"  Precision : {metrics['precision']:.2%}")
    print(f"  F1        : {metrics['f1']:.2%}")
    print(f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  FN={metrics['fn']:,}  TN={metrics['tn']:,}")
    print("═" * 42)


if __name__ == "__main__":
    main()
