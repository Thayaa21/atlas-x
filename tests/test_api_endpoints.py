"""
ATLAS-X API – Endpoint tests
──────────────────────────────
Tests all six v1 endpoints using real feature values from the holdout set.

Run the API first:
    ./run_api.sh

Then in a separate terminal:
    python tests/test_api_endpoints.py
"""
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
SAMPLES_PATH = Path(__file__).parent / "sample_transactions.json"


def _load_samples():
    if not SAMPLES_PATH.exists():
        print("[ERROR] tests/sample_transactions.json not found.")
        print("  Run: python tests/generate_samples.py  (or re-run the test setup)")
        sys.exit(1)
    return json.loads(SAMPLES_PATH.read_text())


# ── Individual tests ──────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE_URL}/api/v1/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["status"] == "healthy", f"Unexpected status: {data['status']}"
    assert data["xgboost_loaded"] is True, "XGBoost not loaded"
    neo4j_status = "connected" if data["neo4j_connected"] else "unavailable (graph disabled)"
    print(f"  [health] OK  |  neo4j={neo4j_status}  |  uptime={data['uptime_seconds']:.1f}s")


def test_predict(samples):
    fraud_sample = next(s for s in samples if s["is_fraud"] == 1)
    legit_sample = next(s for s in samples if s["is_fraud"] == 0)

    for label, sample in [("fraud", fraud_sample), ("legit", legit_sample)]:
        payload = {
            "transaction_id": sample["transaction_id"],
            "card_id":        sample["card_id"],
            "amount":         sample["amount"],
            "features":       sample["features"],
        }
        r = requests.post(f"{BASE_URL}/api/v1/predict", json=payload, timeout=15)
        assert r.status_code == 200, f"[{label}] Expected 200, got {r.status_code}: {r.text}"
        data = r.json()

        required = ["transaction_id", "fraud_probability", "graph_risk_score",
                    "customer_segment", "decision", "confidence", "latency_ms", "timestamp"]
        for field in required:
            assert field in data, f"[{label}] Missing field: {field}"

        assert data["decision"] in ("APPROVE", "FLAG", "BLOCK"), \
            f"[{label}] Unknown decision: {data['decision']}"
        assert 0.0 <= data["fraud_probability"] <= 1.0, \
            f"[{label}] Invalid probability: {data['fraud_probability']}"

        print(f"  [predict/{label:5s}] txn={sample['transaction_id']}  "
              f"prob={data['fraud_probability']:.3f}  "
              f"decision={data['decision']:<8}  "
              f"segment={data['customer_segment']:<9}  "
              f"latency={data['latency_ms']:.1f}ms")


def test_stats():
    r = requests.get(f"{BASE_URL}/api/v1/stats", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "total_predictions" in data
    assert "decisions" in data
    print(f"  [stats]  total={data['total_predictions']}  "
          f"fraud_rate={data['fraud_rate']:.3f}  "
          f"avg_latency={data['avg_latency_ms']:.1f}ms  "
          f"decisions={data['decisions']}")


def test_graph_card(samples):
    card_id = samples[0]["card_id"]
    r = requests.get(f"{BASE_URL}/api/v1/graph/card/{card_id}", timeout=10)
    if r.status_code == 503:
        print(f"  [graph/card] SKIPPED (Neo4j unavailable)")
        return
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["card_id"] == card_id
    assert "in_fraud_ring" in data
    print(f"  [graph/card] card={card_id}  in_ring={data['in_fraud_ring']}  "
          f"ring_type={data['ring_type']}  "
          f"graph_risk={data['graph_risk_score']:.3f}")


def test_graph_rings():
    r = requests.get(f"{BASE_URL}/api/v1/graph/rings?min_cards=3&min_frauds=2",
                     timeout=15)
    if r.status_code == 503:
        print(f"  [graph/rings] SKIPPED (Neo4j unavailable)")
        return
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "rings" in data
    assert "total" in data
    print(f"  [graph/rings] total={data['total']}  "
          f"top_ring={data['rings'][0] if data['rings'] else 'none'}")


def test_explain(samples):
    sample = samples[0]
    payload = {
        "transaction_id": sample["transaction_id"],
        "features":       sample["features"],
    }
    r = requests.post(f"{BASE_URL}/api/v1/explain", json=payload, timeout=60)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "shap_values" in data
    assert len(data["shap_values"]) <= 10
    top_feature = max(data["shap_values"], key=lambda k: abs(data["shap_values"][k]))
    print(f"  [explain]  top_feature={top_feature}  "
          f"shap={data['shap_values'][top_feature]:.4f}  "
          f"reasoning='{data['decision_reasoning']}'")


def test_legacy_healthz():
    r = requests.get(f"{BASE_URL}/healthz", timeout=5)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    print(f"  [/healthz] OK")


def test_latency(samples):
    """Quick 10-request latency benchmark."""
    sample = samples[1]
    payload = {
        "transaction_id": sample["transaction_id"],
        "card_id":        sample["card_id"],
        "amount":         sample["amount"],
        "features":       sample["features"],
    }
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        r  = requests.post(f"{BASE_URL}/api/v1/predict", json=payload, timeout=15)
        latencies.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200

    import statistics
    print(f"  [latency]  avg={statistics.mean(latencies):.1f}ms  "
          f"p50={statistics.median(latencies):.1f}ms  "
          f"max={max(latencies):.1f}ms  (10 requests)")


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    print(f"ATLAS-X API tests → {BASE_URL}\n")

    # Check server is up
    try:
        requests.get(f"{BASE_URL}/healthz", timeout=5)
    except requests.ConnectionError:
        print(f"[ERROR] Cannot connect to {BASE_URL}")
        print("  Make sure the API is running:  ./run_api.sh")
        sys.exit(1)

    samples = _load_samples()
    errors  = []

    tests = [
        ("Health endpoint",       lambda: test_health()),
        ("Legacy /healthz",       lambda: test_legacy_healthz()),
        ("Predict (fraud+legit)", lambda: test_predict(samples)),
        ("Stats endpoint",        lambda: test_stats()),
        ("Graph card lookup",     lambda: test_graph_card(samples)),
        ("Graph rings list",      lambda: test_graph_rings()),
        ("Explain (SHAP)",        lambda: test_explain(samples)),
        ("Latency benchmark",     lambda: test_latency(samples)),
    ]

    for name, fn in tests:
        print(f"\n{name}")
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            errors.append((name, str(e)))
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            errors.append((name, str(e)))

    print("\n" + "=" * 52)
    if errors:
        print(f"  {len(errors)} test(s) FAILED:")
        for name, msg in errors:
            print(f"    ✗ {name}: {msg}")
    else:
        print(f"  All {len(tests)} tests PASSED")
    print("=" * 52)


if __name__ == "__main__":
    main()
