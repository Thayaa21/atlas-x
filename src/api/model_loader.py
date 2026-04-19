"""
ATLAS-X API – Singleton model loader.

Loads once at startup:
  • XGBoost v4 classifier  (src/models/atlass_x_xgb_v4_graph.pkl)
  • Segment threshold artifact (src/optimization/artifacts/thresholds.json)
  • Feature dtype map  (inferred from training parquet at first load)
  • SHAP TreeExplainer  (lazy — built on first /explain request)
  • Neo4j FraudRingChecker (optional — degrades gracefully if Neo4j is down)
"""
import json
import threading
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

REPO       = Path(__file__).resolve().parents[2]
V4_MODEL   = REPO / "src/models/atlass_x_xgb_v4_graph.pkl"
THRESH_P   = REPO / "src/optimization/artifacts/thresholds.json"
V4_DATA    = REPO / "data/processed/train_with_graph_features.parquet"

GRAPH_COLS = [
    "device_fraud_rate", "device_card_velocity", "connected_fraud_cards",
    "email_fraud_rate", "address_fraud_rate", "graph_risk_score",
]

DROP_COLS = ["isFraud", "TransactionID", "TransactionDT", "customer_segment"]


class ModelLoader:
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded = False
        return cls._instance

    # ── Public interface ──────────────────────────────────────────────────────

    def load_all(self) -> None:
        if self._loaded:
            return

        print("[Loader] Loading XGBoost v4...")
        self.model = joblib.load(V4_MODEL)
        self.feature_names: list[str] = list(self.model.feature_names_in_)

        print("[Loader] Loading threshold artifact...")
        thresh = json.loads(THRESH_P.read_text())
        self.thresholds: Dict[str, float] = {
            s: float(thresh["segments"][s]["threshold"])
            for s in ["VIP", "Regular", "New"]
        }
        self.seg_params: dict = thresh.get("segment_quantile_cuts", {})

        print("[Loader] Inferring feature dtypes from training parquet (sample)...")
        self._cat_cols = self._infer_cat_cols()

        self._shap_explainer = None          # lazy init
        self._shap_lock      = threading.Lock()

        print("[Loader] Connecting to Neo4j (optional)...")
        try:
            from src.graph.fraud_ring_checker import FraudRingChecker
            self.ring_checker: Optional[object] = FraudRingChecker.get()
            self._neo4j_ok = True
            print("[Loader] Neo4j connected.")
        except Exception as exc:
            print(f"[Loader] Neo4j unavailable: {exc}")
            self.ring_checker = None
            self._neo4j_ok = False

        self._loaded = True
        print("[Loader] Ready.")

    @property
    def neo4j_ok(self) -> bool:
        return getattr(self, "_neo4j_ok", False)

    # ── Feature preparation ───────────────────────────────────────────────────

    def prepare_features(self, features: dict) -> pd.DataFrame:
        """
        Build a single-row DataFrame matching the model's expected feature matrix.

        The caller passes a flat dict with pre-computed feature values (same set
        used during training).  Categorical columns are cast to `category` dtype.
        Missing features default to 0 / "None".
        """
        row: dict = {}
        for col in self.feature_names:
            val = features.get(col)
            if col in self._cat_cols:
                row[col] = str(val) if val is not None else "None"
            else:
                try:
                    row[col] = float(val) if val is not None else 0.0
                except (TypeError, ValueError):
                    row[col] = 0.0

        df = pd.DataFrame([row])
        for col in self._cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        for col in GRAPH_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(float)
        return df

    # ── Segment inference ─────────────────────────────────────────────────────

    def segment_for(self, features: dict) -> str:
        """
        Infer customer segment from D1/D15 using persisted quantile cut params.
        Falls back to 'Regular' if the seg_params artifact is missing.
        """
        if not self.seg_params:
            return "Regular"
        try:
            eps     = 1e-12
            d1      = float(features.get("D1") or 0.0)
            d15     = float(features.get("D15") or 0.0)
            d1_min  = float(self.seg_params["d1_min"])
            d1_max  = float(self.seg_params["d1_max"])
            d15_min = float(self.seg_params["d15_min"])
            d15_max = float(self.seg_params["d15_max"])
            vip_cut = float(self.seg_params["vip_cut"])
            reg_cut = float(self.seg_params["regular_cut"])

            tenure_score  = (d1  - d1_min)  / (d1_max  - d1_min  + eps)
            recency_score = (d15_max - d15) / (d15_max  - d15_min + eps)
            combined      = 0.5 * tenure_score + 0.5 * recency_score

            if combined >= vip_cut:
                return "VIP"
            if combined >= reg_cut:
                return "Regular"
            return "New"
        except Exception:
            return "Regular"

    # ── SHAP explainer (lazy) ─────────────────────────────────────────────────

    def get_shap_explainer(self):
        if self._shap_explainer is None:
            with self._shap_lock:
                if self._shap_explainer is None:
                    print("[Loader] Building SHAP TreeExplainer (first call only)...")
                    import shap
                    self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_cat_cols(self) -> set:
        """
        Read the first 1 000 rows of the training parquet to identify which
        columns are object/category dtype → those need `category` cast at
        inference time.
        """
        try:
            sample = pd.read_parquet(V4_DATA, columns=self.feature_names[:50])  # partial read
            # read all feature columns in one shot for dtype inference
            sample = pd.read_parquet(V4_DATA).head(500)
            cat_cols = set(
                sample.select_dtypes(include=["object", "category"]).columns
            ) & set(self.feature_names)
            return cat_cols
        except Exception:
            # Fallback: known categorical columns from the original dataset
            return {
                "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
                "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
                "DeviceType", "id_12", "id_15", "id_16", "id_23", "id_27",
                "id_28", "id_29", "id_30", "id_31", "id_33", "id_34",
                "id_35", "id_36", "id_37", "id_38",
            }
