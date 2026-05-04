import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ThresholdTargets:
    precision_min: float = 0.85
    recall_min: float = 0.75


SEGMENTS: List[str] = ["VIP", "Regular", "New"]

# ATLAS-X cost matrix (kept consistent with `src/models/optimize_threshold.py`)
COST_FN = 2000  # missed a fraudster
COST_FP = 50  # annoyed a legitimate customer


def _minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo + eps)


def compute_customer_segments(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    """
    VIP/Regular/New segmentation driven by `tenure_recency`.

    VIP definition (per requirements):
    - long tenure => high D1
    - recent activity => low D15

    We compute a combined score = 0.5*tenure_score + 0.5*recency_score,
    where tenure_score is min-max of D1, and recency_score is min-max of (-D15).
    Then we split by quantiles:
      - VIP: top 10%
      - Regular: next 40%
      - New: remaining 60%
    """
    if "D1" not in df.columns or "D15" not in df.columns:
        raise ValueError("Expected columns `D1` and `D15` for segmentation.")

    d1 = df["D1"].to_numpy().astype(float)
    d15 = df["D15"].to_numpy().astype(float)

    # Store normalization stats so we can reproduce online segmentation.
    d1_min = float(np.nanmin(d1))
    d1_max = float(np.nanmax(d1))
    d15_min = float(np.nanmin(d15))
    d15_max = float(np.nanmax(d15))

    eps = 1e-12
    tenure_score = (d1 - d1_min) / (d1_max - d1_min + eps)

    # recency_score uses min-max(-D15); equivalently: (d15_max - D15) / (d15_max - d15_min)
    recency_score = (d15_max - d15) / (d15_max - d15_min + eps)
    combined = 0.5 * tenure_score + 0.5 * recency_score

    vip_cut = np.nanquantile(combined, 0.90)  # top 10%
    regular_cut = np.nanquantile(combined, 0.50)  # bottom 50 -> VIP+Regular

    # VIP >= vip_cut
    # Regular in [regular_cut, vip_cut)
    # New < regular_cut
    segment = np.where(combined >= vip_cut, "VIP", np.where(combined >= regular_cut, "Regular", "New"))
    segment_s = pd.Series(segment, index=df.index, name="customer_segment")

    cuts = {
        "vip_cut": float(vip_cut),
        "regular_cut": float(regular_cut),
        "d1_min": d1_min,
        "d1_max": d1_max,
        "d15_min": d15_min,
        "d15_max": d15_max,
    }
    return segment_s, cuts


def _business_cost(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """
    ATLAS-X business loss:
      - FN cost: COST_FN for y_true=1 but predicted 0
      - FP cost: COST_FP for y_true=0 but predicted 1
    """
    tp = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    return float(fn * COST_FN + fp * COST_FP), {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def choose_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    targets: ThresholdTargets,
    thresholds: np.ndarray,
) -> Dict:
    best = None
    best_cost = None
    best_f1 = None

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        # Handle edge cases (precision/recall undefined when no positives predicted/true)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if prec + 1e-12 < targets.precision_min or rec + 1e-12 < targets.recall_min:
            continue

        cost, counts = _business_cost(y_true, y_pred)

        # Prefer lower business cost; tie-break by higher F1
        denom = (prec + rec + 1e-12)
        f1 = float(2 * prec * rec / denom)

        candidate = {
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "business_cost": float(cost),
            "confusion_counts": counts,
        }

        if best is None or cost < best_cost - 1e-6 or (abs(cost - best_cost) <= 1e-6 and f1 > best_f1):
            best = candidate
            best_cost = cost
            best_f1 = f1

    if best is not None:
        return {"selected": True, **best}

    # Fallback: maximize F1 with penalty if precision/recall constraints fail.
    best = None
    best_score = -np.inf
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        # score: reward recall, reward precision, but penalize missing constraints
        penalty_prec = max(0.0, targets.precision_min - prec)
        penalty_rec = max(0.0, targets.recall_min - rec)
        f1_denom = (prec + rec + 1e-12)
        f1 = float(2 * prec * rec / f1_denom)
        score = f1 - 3.0 * (penalty_prec + penalty_rec)

        if score > best_score:
            best_score = score
            cost, counts = _business_cost(y_true, y_pred)
            best = {
                "threshold": float(t),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "business_cost": float(cost),
                "confusion_counts": counts,
            }

    return {"selected": False, **best}


def optimize_thresholds(
    *,
    model_path: Path,
    data_path: Path,
    out_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold_grid: Tuple[float, float, int] = (0.01, 0.99, 99),
    targets: ThresholdTargets = ThresholdTargets(),
) -> Dict:
    df = pd.read_parquet(data_path)
    if "isFraud" not in df.columns:
        raise ValueError("Expected target column `isFraud`.")

    # Segmenting is based on D1/D15 and must happen before dropping columns.
    segment_s, cuts = compute_customer_segments(df)
    df = df.copy()
    df["customer_segment"] = segment_s

    feature_cols = [
        c
        for c in df.columns
        if c not in ("isFraud", "TransactionID", "TransactionDT", "customer_segment")
    ]
    # Copy to avoid SettingWithCopyWarning when we later cast categoricals.
    X = df.loc[:, feature_cols].copy()
    y = df["isFraud"].astype(int).to_numpy()

    # We rely on the model being trained with the same feature set.
    # XGBoost categorical support requires category dtypes; we mirror the training behavior.
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("object").fillna("None").astype("category")

    X_train, X_val, y_train, y_val, seg_train, seg_val = train_test_split(
        X, y, df["customer_segment"], test_size=test_size, random_state=random_state, stratify=y
    )

    model = joblib.load(model_path)
    # Note: using predict_proba on validation set to choose thresholds.
    val_probs = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(threshold_grid[0], threshold_grid[1], threshold_grid[2], dtype=float)

    results = {
        "targets": targets.__dict__,
        "segment_quantile_cuts": cuts,
        "segments": {},
        "metadata": {
            "model_path": str(model_path),
            "data_path": str(data_path),
            "random_state": random_state,
            "test_size": test_size,
            "threshold_grid": {"min": threshold_grid[0], "max": threshold_grid[1], "num": threshold_grid[2]},
        },
    }

    for seg in SEGMENTS:
        mask = seg_val == seg
        if int(mask.sum()) == 0:
            results["segments"][seg] = {"error": "no samples in segment"}
            continue
        seg_y = y_val[mask.values]
        seg_p = val_probs[mask.values]
        res = choose_best_threshold(seg_y, seg_p, targets, thresholds)
        results["segments"][seg] = res

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


def load_threshold_artifact(path: Path) -> Dict:
    return json.loads(path.read_text())


def select_segment_threshold(threshold_artifact: Dict, segment: str) -> float:
    return float(threshold_artifact["segments"][segment]["threshold"])


if __name__ == "__main__":
    # Default offline calibration using existing v2 model and clustered parquet.
    # We intentionally keep this script self-contained so the online services can consume the produced JSON.
    # File: src/optimization/threshold_optimizer.py
    # parents[0] = src/optimization
    # parents[1] = src
    # parents[2] = repo root
    repo_root = Path(__file__).resolve().parents[2]

    # Prefer calibrating on the v3 production pipeline artifacts.
    model_p_v3 = repo_root / "src/models/atlass_x_xgb_v3.pkl"
    data_p_full = repo_root / "data/processed/train_full_features.parquet"

    if model_p_v3.exists() and data_p_full.exists():
        model_p = model_p_v3
        data_p = data_p_full
    else:
        # Fallback: calibrate on baseline v2 artifacts so the script still runs
        # during early development. Once v3 is trained, we overwrite thresholds.
        model_p = repo_root / "src/models/atlass_x_xgb_v2.pkl"
        data_p = repo_root / "data/processed/train_clustered.parquet"
        print("Warning: v3 artifacts not found; calibrating thresholds on v2 baseline.")
    out_p = repo_root / "src/optimization/artifacts/thresholds.json"

    optimize_thresholds(
        model_path=model_p,
        data_path=data_p,
        out_path=out_p,
    )

