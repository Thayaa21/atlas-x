"""
ATLAS-X – Kafka Data Augmentor (Feature 9)
───────────────────────────────────────────
Generates synthetic fraud transactions to augment the Kafka stream.

Class: FraudAugmentor
Methods:
  generate_synthetic_fraud()   – create a fake fraud using SMOTE-like logic
  inject_noise()               – add Gaussian noise to a real transaction
  create_fraud_variants()      – slightly modify existing fraud transactions

Fraud patterns generated:
  high_amount    – large transaction amount with a new account
  device_sharing – multiple cards from the same device
  rapid_sequence – rapid back-to-back transactions
  cross_border   – international + domestic transaction same day
  round_number   – round-dollar amounts (money laundering pattern)

Usage (standalone):
    from src.streaming.data_augmentor import FraudAugmentor
    aug = FraudAugmentor(real_transactions)
    synthetic = aug.generate_synthetic_fraud(n=200)
"""
import math
import random
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Pattern registry ──────────────────────────────────────────────────────────

FRAUD_PATTERNS = [
    "high_amount",
    "device_sharing",
    "rapid_sequence",
    "cross_border",
    "round_number",
]

# Numeric feature columns that can safely receive Gaussian noise
_NUMERIC_COLS = [
    "TransactionAmt", "dist1", "dist2",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
]

# Categorical columns that identify devices / emails / addresses
_DEVICE_COLS  = ["DeviceInfo", "id_31"]
_EMAIL_COLS   = ["P_emaildomain", "R_emaildomain"]
_COUNTRY_COLS = ["addr2"]


@dataclass
class AugmentationStats:
    total_generated: int = 0
    by_pattern: dict = field(default_factory=lambda: {p: 0 for p in FRAUD_PATTERNS})

    def log(self) -> str:
        parts = ", ".join(f"{k}: {v}" for k, v in self.by_pattern.items() if v > 0)
        return (
            f"Generated {self.total_generated} synthetic frauds "
            f"(types: {parts})"
        )


class FraudAugmentor:
    """
    Generates synthetic fraud transactions by mutating real examples.

    Parameters
    ----------
    real_transactions:
        List of message dicts in the format produced by
        ``kafka_producer.load_holdout()``.  Only fraud transactions
        (``is_fraud == 1``) are used as seeds.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        real_transactions: list[dict],
        *,
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)

        self._all   = real_transactions
        self._frauds = [t for t in real_transactions if t.get("is_fraud") == 1]
        self._legit  = [t for t in real_transactions if t.get("is_fraud") == 0]

        if not self._frauds:
            raise ValueError(
                "No fraud transactions found in the provided dataset. "
                "Ensure messages include 'is_fraud': 1 for fraud rows."
            )

        self.stats = AugmentationStats()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_synthetic_fraud(self, n: int = 1) -> list[dict]:
        """
        Generate *n* synthetic fraud transactions using SMOTE-like interpolation.

        Each synthetic transaction is created by:
          1. Picking a random real fraud as the seed.
          2. Picking a random fraud pattern.
          3. Applying the pattern-specific mutation.
          4. Tagging the message with synthetic metadata.
        """
        results: list[dict] = []
        for _ in range(n):
            seed_txn = random.choice(self._frauds)
            pattern  = random.choice(FRAUD_PATTERNS)
            mutator  = getattr(self, f"_pattern_{pattern}")
            synthetic = mutator(deepcopy(seed_txn))
            synthetic["is_fraud"] = 1
            synthetic["metadata"] = {
                "synthetic": True,
                "pattern":   pattern,
                "seed_id":   seed_txn.get("transaction_id"),
            }
            results.append(synthetic)
            self.stats.total_generated += 1
            self.stats.by_pattern[pattern] += 1
        return results

    def inject_noise(self, transaction: dict, *, noise_std: float = 0.05) -> dict:
        """
        Add Gaussian noise (relative, std=*noise_std*) to numeric features.
        Useful for creating near-duplicate variants of real transactions.
        """
        txn = deepcopy(transaction)
        feats = txn.get("features", {})
        for col in _NUMERIC_COLS:
            if col in feats and feats[col] is not None:
                val = float(feats[col])
                if val != 0.0:
                    feats[col] = round(
                        val * (1.0 + np.random.normal(0, noise_std)), 4
                    )
        txn["features"] = feats
        txn["transaction_id"] = f"noise-{uuid.uuid4().hex[:8]}"
        return txn

    def create_fraud_variants(self, n: int = 1) -> list[dict]:
        """
        Create *n* slightly-modified variants of existing fraud transactions
        by injecting small noise.  Useful for stress-testing the model.
        """
        results: list[dict] = []
        for _ in range(n):
            seed = random.choice(self._frauds)
            variant = self.inject_noise(seed, noise_std=0.03)
            variant["is_fraud"] = 1
            variant["metadata"] = {
                "synthetic": True,
                "pattern":   "variant",
                "seed_id":   seed.get("transaction_id"),
            }
            results.append(variant)
        return results

    # ── Pattern mutators ──────────────────────────────────────────────────────

    def _new_txn_id(self) -> str:
        return f"syn-{uuid.uuid4().hex[:12]}"

    def _pattern_high_amount(self, txn: dict) -> dict:
        """Large transaction amount with a new account (D1 ≈ 0)."""
        feats = txn.get("features", {})
        # Multiply amount by 5–20×
        amt = float(feats.get("TransactionAmt") or 100.0)
        feats["TransactionAmt"] = round(amt * random.uniform(5.0, 20.0), 2)
        txn["amount"] = feats["TransactionAmt"]
        # New account: D1 (days since first transaction) near 0
        feats["D1"]  = round(random.uniform(0.0, 3.0), 1)
        feats["D15"] = round(random.uniform(0.0, 5.0), 1)
        # Boost graph risk
        feats["graph_risk_score"] = round(random.uniform(0.6, 0.95), 4)
        txn["features"]       = feats
        txn["transaction_id"] = self._new_txn_id()
        return txn

    def _pattern_device_sharing(self, txn: dict) -> dict:
        """Multiple cards from the same device – high device_card_velocity."""
        feats = txn.get("features", {})
        feats["device_card_velocity"]  = random.randint(5, 20)
        feats["device_fraud_rate"]     = round(random.uniform(0.4, 0.9), 4)
        feats["connected_fraud_cards"] = random.randint(3, 10)
        feats["graph_risk_score"]      = round(random.uniform(0.65, 0.95), 4)
        # Assign a shared device fingerprint
        feats["DeviceInfo"] = f"shared_device_{random.randint(1, 50)}"
        txn["features"]       = feats
        txn["transaction_id"] = self._new_txn_id()
        return txn

    def _pattern_rapid_sequence(self, txn: dict) -> dict:
        """Rapid back-to-back transactions – C1/C2 velocity counters high."""
        feats = txn.get("features", {})
        # C1–C4 are card-level velocity counters
        for c in ["C1", "C2", "C3", "C4"]:
            feats[c] = random.randint(10, 50)
        # D9 = hours since last transaction (very recent)
        feats["D9"] = round(random.uniform(0.0, 0.5), 3)
        feats["graph_risk_score"] = round(random.uniform(0.5, 0.85), 4)
        txn["features"]       = feats
        txn["transaction_id"] = self._new_txn_id()
        return txn

    def _pattern_cross_border(self, txn: dict) -> dict:
        """International + domestic transaction same day."""
        feats = txn.get("features", {})
        # addr2 encodes country code; set to a foreign country
        feats["addr2"] = random.choice([
            "87.0", "32.0", "45.0", "60.0", "96.0"
        ])
        # dist2 = distance to billing address (large for cross-border)
        feats["dist2"] = round(random.uniform(5000.0, 15000.0), 1)
        feats["dist1"] = round(random.uniform(0.0, 50.0), 1)
        feats["graph_risk_score"] = round(random.uniform(0.45, 0.80), 4)
        txn["features"]       = feats
        txn["transaction_id"] = self._new_txn_id()
        return txn

    def _pattern_round_number(self, txn: dict) -> dict:
        """Round-dollar amounts – common money laundering pattern."""
        feats = txn.get("features", {})
        # Pick a round amount: 100, 500, 1000, 5000, 10000
        round_amounts = [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        amt = random.choice(round_amounts)
        feats["TransactionAmt"] = amt
        txn["amount"] = amt
        # Multiple round-number transactions → high C14 (count of transactions)
        feats["C14"] = random.randint(5, 30)
        feats["graph_risk_score"] = round(random.uniform(0.4, 0.75), 4)
        txn["features"]       = feats
        txn["transaction_id"] = self._new_txn_id()
        return txn


# ── Convenience: mix real + synthetic ────────────────────────────────────────

def augment_batch(
    real_messages: list[dict],
    *,
    synthetic_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], AugmentationStats]:
    """
    Return a mixed batch of real + synthetic transactions.

    Parameters
    ----------
    real_messages:
        Original messages from ``load_holdout()``.
    synthetic_ratio:
        Fraction of the output that should be synthetic (default 0.20 = 20%).
    seed:
        Random seed.

    Returns
    -------
    (mixed_messages, stats)
    """
    aug = FraudAugmentor(real_messages, seed=seed)
    n_synthetic = max(1, int(len(real_messages) * synthetic_ratio))
    synthetic   = aug.generate_synthetic_fraud(n=n_synthetic)

    mixed = real_messages + synthetic
    random.shuffle(mixed)
    return mixed, aug.stats
