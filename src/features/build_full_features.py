import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _safe_abs(s: pd.Series) -> pd.Series:
    return s.astype(float).abs()


def _compute_features(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    eps = 1e-6

    # Fill numeric missing values for deterministic feature math.
    for col in ["addr2", "dist1", "dist2"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # --- Velocity (15) ---
    vel_uid_count_log1p = np.log1p(df["uid_count"].astype(float).fillna(0))
    vel_relative_amt = df["uid_Amt_Relative_Check"].astype(float).fillna(0)
    vel_relative_absdev = (vel_relative_amt - 1.0).abs()
    vel_relative_sq = vel_relative_amt**2
    vel_relative_log = np.log1p(vel_relative_absdev)

    df["vel_uid_count_log1p"] = vel_uid_count_log1p
    df["vel_uid_count_div_time"] = vel_uid_count_log1p / (1.0 + df["Transaction_Hour"].astype(float) + df["Transaction_Day"].astype(float))
    df["vel_relative_amt"] = vel_relative_amt
    df["vel_relative_amt_sq"] = vel_relative_sq
    df["vel_relative_amt_log"] = vel_relative_log
    df["vel_relative_amt_absdev"] = vel_relative_absdev
    df["vel_amt_per_uid_count"] = df["TransactionAmt"].astype(float) / (df["uid_count"].astype(float).fillna(0) + 1.0)
    df["vel_amt_log_per_uid_count"] = df["TransactionAmt_Log"].astype(float) / (vel_uid_count_log1p + 1.0)

    vel_recency_factor = 1.0 / (1.0 + df["D15"].astype(float) + eps)
    vel_tenure_factor = 1.0 / (1.0 + df["D1"].astype(float) + eps)

    df["vel_recency_factor"] = vel_recency_factor
    df["vel_tenure_factor"] = vel_tenure_factor
    df["vel_recency_x_uid_count"] = vel_recency_factor * vel_uid_count_log1p
    df["vel_recency_x_relative"] = vel_recency_factor * vel_relative_amt
    df["vel_hour_x_uid_count"] = df["Transaction_Hour"].astype(float) * vel_uid_count_log1p
    df["vel_day_x_relative"] = (df["Transaction_Day"].astype(float) + 1.0) * vel_relative_amt
    df["vel_hour_day_interaction"] = (df["Transaction_Hour"].astype(float) + 1.0) * (df["Transaction_Day"].astype(float) + 1.0)

    # --- Geographic (15) ---
    addr1_abs = _safe_abs(df["addr1"])
    addr2_abs = _safe_abs(df.get("addr2", pd.Series(np.full(len(df), -1), index=df.index)))
    dist1_abs = _safe_abs(df["dist1"]) if "dist1" in df.columns else pd.Series(np.full(len(df), -1), index=df.index)
    dist2_abs = _safe_abs(df["dist2"]) if "dist2" in df.columns else pd.Series(np.full(len(df), -1), index=df.index)

    geo_addr1_log = np.log1p(addr1_abs + eps)
    geo_addr1_bucket = np.floor(addr1_abs / 10.0)

    geo_addr2_log = np.log1p(addr2_abs + eps)
    geo_dist1_abs = dist1_abs
    geo_dist2_abs = dist2_abs
    geo_dist1_log = np.log1p(geo_dist1_abs + eps)
    geo_dist2_log = np.log1p(geo_dist2_abs + eps)

    geo_dist_total = geo_dist1_abs + geo_dist2_abs
    geo_dist_total_log = np.log1p(geo_dist_total + eps)
    geo_dist_ratio = geo_dist1_abs / (geo_dist2_abs + 1.0)
    geo_dist_symmetry = 1.0 / (1.0 + (geo_dist1_abs - geo_dist2_abs).abs())

    df["geo_addr1_log"] = geo_addr1_log
    df["geo_addr1_bucket"] = geo_addr1_bucket
    df["geo_addr2_log"] = geo_addr2_log
    df["geo_dist1_abs"] = geo_dist1_abs
    df["geo_dist2_abs"] = geo_dist2_abs
    df["geo_dist1_log"] = geo_dist1_log
    df["geo_dist2_log"] = geo_dist2_log
    df["geo_dist_total"] = geo_dist_total
    df["geo_dist_total_log"] = geo_dist_total_log
    df["geo_dist_ratio"] = geo_dist_ratio
    df["geo_addr1_x_dist1_log"] = geo_addr1_log * geo_dist1_log
    df["geo_addr1_x_dist2_log"] = geo_addr1_log * geo_dist2_log
    df["geo_dist_symmetry"] = geo_dist_symmetry
    df["geo_geo_interaction"] = (geo_addr1_bucket + 1.0) * (geo_dist_total_log + 1.0)
    df["geo_addr_addr2_interaction"] = (addr1_abs + 1.0) * (addr2_abs + 1.0) / (1.0 + geo_dist_total)

    # --- Temporal (10) ---
    hour = df["Transaction_Hour"].astype(float)
    day = df["Transaction_Day"].astype(float)
    hour_rad = 2.0 * np.pi * hour / 24.0
    day_rad = 2.0 * np.pi * day / 7.0

    temp_hour_sin = np.sin(hour_rad)
    temp_hour_cos = np.cos(hour_rad)
    temp_day_sin = np.sin(day_rad)
    temp_day_cos = np.cos(day_rad)

    temp_recency_factor = vel_recency_factor  # same definition
    temp_tenure_factor = vel_tenure_factor

    df["temp_hour_sin"] = temp_hour_sin
    df["temp_hour_cos"] = temp_hour_cos
    df["temp_day_sin"] = temp_day_sin
    df["temp_day_cos"] = temp_day_cos
    df["temp_recency_factor"] = temp_recency_factor
    df["temp_tenure_factor"] = temp_tenure_factor
    df["temp_D1_over_D15"] = df["D1"].astype(float) / (df["D15"].astype(float) + 1.0 + eps)
    df["temp_recency_x_hour"] = temp_recency_factor * (hour + 1.0)
    df["temp_tenure_x_day"] = temp_tenure_factor * (day + 1.0)
    df["temp_hour_bucket"] = np.floor(hour / 6.0)

    # --- Consistency (10) ---
    global_fraud_rate = float(params["global_fraud_rate"])
    rel_high_cut = float(params["relative_amt_high_cut"])
    d15_low_cut = float(params["d15_recent_low_cut"])

    cons_relative_absdev = vel_relative_absdev
    df["cons_relative_absdev"] = cons_relative_absdev
    df["cons_relative_logdev"] = np.log1p(cons_relative_absdev)
    df["cons_cluster_risk_x_relative"] = df["cluster_fraud_rate"].astype(float) * vel_relative_amt
    df["cons_cluster_risk_x_recency"] = df["cluster_fraud_rate"].astype(float) * temp_recency_factor
    df["cons_cluster_minus_global"] = df["cluster_fraud_rate"].astype(float) - global_fraud_rate
    df["cons_uid_count_x_cluster"] = vel_uid_count_log1p * df["cluster_fraud_rate"].astype(float)
    df["cons_uid_count_over_day"] = df["uid_count"].astype(float) / (day + 1.0)
    df["cons_high_risk_flag"] = (
        (vel_relative_amt >= rel_high_cut) & (df["D15"].astype(float) <= d15_low_cut)
    ).astype(int)
    df["cons_mismatch_score"] = (df["cluster_fraud_rate"].astype(float) - 0.5) * (vel_relative_amt - 1.0)
    df["cons_transaction_amt_log_x_cluster"] = df["TransactionAmt_Log"].astype(float) * df["cluster_fraud_rate"].astype(float)

    return df


def build_full_features(
    *,
    input_parquet: Path,
    output_parquet: Path,
    params_out_path: Path,
) -> None:
    df = pd.read_parquet(input_parquet)

    # Parameters used for consistent feature generation (especially flag thresholds).
    params = {
        "global_fraud_rate": float(df["isFraud"].mean()),
        # High relative amount threshold (top quartile)
        "relative_amt_high_cut": float(df["uid_Amt_Relative_Check"].quantile(0.75)),
        # Recent activity threshold (bottom quartile of D15; smaller D15 = more recent)
        "d15_recent_low_cut": float(df["D15"].quantile(0.25)),
    }

    df_out = _compute_features(df.copy(), params)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_parquet, engine="pyarrow")

    params_out_path.parent.mkdir(parents=True, exist_ok=True)
    params_out_path.write_text(json.dumps(params, indent=2))

    # Minimal sanity check: we expect +50 columns.
    new_feature_cols = [
        c
        for c in df_out.columns
        if c.startswith(("vel_", "geo_", "temp_", "cons_"))
    ]
    print(f"Built full features -> {output_parquet}")
    print(f"Added derived feature count: {len(new_feature_cols)} (expected 50)")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    in_p = repo_root / "data/processed/train_clustered.parquet"
    out_p = repo_root / "data/processed/train_full_features.parquet"
    params_p = repo_root / "src/features/artifacts/full_feature_params.json"

    build_full_features(
        input_parquet=in_p,
        output_parquet=out_p,
        params_out_path=params_p,
    )

