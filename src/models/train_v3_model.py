import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    auc as _unused_auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.optimization.threshold_optimizer import compute_customer_segments


def _safe_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_with_thresholds(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    segments: pd.Series,
    segment_thresholds: Dict[str, float],
    default_threshold: float = 0.5,
) -> Dict:
    preds = np.zeros_like(y_prob, dtype=int)
    for seg in ["VIP", "Regular", "New"]:
        mask = segments == seg
        t = float(segment_thresholds.get(seg, default_threshold))
        preds[mask.values] = (y_prob[mask.values] >= t).astype(int)

    overall = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        **_safe_precision_recall_f1(y_true, preds),
    }

    per_segment = {}
    for seg in ["VIP", "Regular", "New"]:
        mask = segments == seg
        if int(mask.sum()) == 0:
            per_segment[seg] = {"error": "no samples"}
            continue
        seg_y = y_true[mask.values]
        seg_preds = preds[mask.values]
        seg_prob = y_prob[mask.values]
        per_segment[seg] = {
            "roc_auc": float(roc_auc_score(seg_y, seg_prob)) if seg_y.min() != seg_y.max() else None,
            "confusion_matrix": confusion_matrix(seg_y, seg_preds).tolist(),
            **_safe_precision_recall_f1(seg_y, seg_preds),
        }

    return {"overall": overall, "per_segment": per_segment}


def train_v3_model(
    *,
    data_path: Path,
    model_out_path: Path,
    threshold_artifact_path: Path | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict:
    df = pd.read_parquet(data_path)
    if "isFraud" not in df.columns:
        raise ValueError("Expected `isFraud` target column.")

    segment_s, _cuts = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = segment_s

    # Drop non-predictive identifiers
    X = df.drop(["isFraud", "TransactionID", "TransactionDT", "customer_segment"], axis=1)
    y = df["isFraud"].astype(int).to_numpy()

    # Cast categoricals (consistent with existing training scripts)
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    # Split
    X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
        X,
        y,
        df["customer_segment"],
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    fraud_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=fraud_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    segment_thresholds: Dict[str, float] = {"VIP": 0.5, "Regular": 0.5, "New": 0.5}

    if threshold_artifact_path is not None and threshold_artifact_path.exists():
        artifact = json.loads(threshold_artifact_path.read_text())
        for seg in ["VIP", "Regular", "New"]:
            segment_thresholds[seg] = float(artifact["segments"][seg]["threshold"])

    metrics = evaluate_with_thresholds(
        y_true=y_test,
        y_prob=y_prob,
        segments=seg_test,
        segment_thresholds=segment_thresholds,
        default_threshold=0.5,
    )

    metrics_out = model_out_path.with_suffix(".metrics.json")
    metrics_out.write_text(json.dumps(metrics, indent=2))
    metrics_out_dict = json.loads(metrics_out.read_text())
    metrics_out_dict["thresholds_used"] = segment_thresholds
    metrics_out.write_text(json.dumps(metrics_out_dict, indent=2))

    return {"model_path": str(model_out_path), "metrics_path": str(metrics_out)}


if __name__ == "__main__":
    # File: src/models/train_v3_model.py
    # parents[0] = src/models
    # parents[1] = src
    # parents[2] = repo root
    repo_root = Path(__file__).resolve().parents[2]
    data_p = repo_root / "data/processed/train_full_features.parquet"
    model_p = repo_root / "src/models/atlass_x_xgb_v3.pkl"
    thresh_p = repo_root / "src/optimization/artifacts/thresholds.json"

    res = train_v3_model(
        data_path=data_p,
        model_out_path=model_p,
        threshold_artifact_path=thresh_p,
    )
    print(f"Model saved: {res['model_path']}")
    print(f"Metrics saved: {res['metrics_path']}")

