# ATLAS-X: Business Overview

## What Is ATLAS-X?

ATLAS-X is a real-time fraud detection platform for financial transactions. It
automatically scores every transaction the moment it happens — in about 27 milliseconds
— and decides whether to approve it, flag it for human review, or block it outright.

The system was built on the IEEE-CIS fraud dataset: 590,540 real-world financial
transactions from a major payment processor, of which 3.5% were confirmed fraud.

---

## The Problem It Solves

Financial fraud is a massive and growing problem. In 2024, global card fraud losses
reached $33.4 billion (Nilson Report). For every $1 of actual fraud, merchants lose
up to $4.60 when you factor in chargebacks, administrative costs, lost goods, and fees
(Risk Solutions, 2024).

But fraud detection is not just about catching fraudsters. It has two equally important
failure modes:

**Missing fraud (False Negatives):**
A transaction slips through that is actually fraudulent. The merchant loses the
transaction amount plus chargeback fees, dispute costs, and operational overhead.
Average total cost: ~$709 per missed fraud transaction (based on avg $154 transaction
× 4.60 multiplier from this dataset).

**Blocking good customers (False Positives):**
A legitimate transaction is wrongly declined. The customer is frustrated, the sale is
lost, and 33% of falsely declined customers reduce or stop spending with that merchant
entirely (Aite-Novarica Group, 2024). Average cost: ~$219 per wrongly blocked
transaction (avg $134 × 1.63 revenue impact multiplier).

Most fraud systems are tuned for high precision — they only block when very confident.
This protects customer experience but leaves a lot of fraud uncaught. ATLAS-X includes
a financial optimiser that finds the threshold that maximises actual profit, not just
model accuracy.

---

## How It Works (Non-Technical)

### Step 1: Transaction Arrives
A customer makes a purchase. The transaction details — amount, card info, device,
email, location — are sent to ATLAS-X in real time.

### Step 2: Two Signals Are Combined
**Signal 1 — The Machine Learning Model:**
ATLAS-X uses a model trained on 590,540 historical transactions. It has learned
patterns that distinguish fraud from legitimate purchases: unusual amounts for a new
account, a device that has been used by multiple cards, a billing address that doesn't
match the card's history. The model produces a fraud probability score between 0 and 1.

**Signal 2 — The Fraud Network Graph:**
ATLAS-X maintains a graph database (Neo4j) that maps relationships between cards,
devices, email addresses, and billing addresses. If a card shares a device with 5 other
cards that have committed fraud, that's a strong signal — even if the model score alone
is borderline. This catches "fraud rings": organised groups using shared infrastructure.

### Step 3: A Decision Is Made
Based on the combined score and a configurable threshold:
- **APPROVE** — low risk, transaction goes through
- **FLAG** — borderline or connected to a fraud ring, sent to a human analyst for review
- **BLOCK** — high confidence fraud, transaction is stopped

### Step 4: Everything Is Recorded
Every decision is stored with a full audit trail. Analysts can see exactly why a
transaction was flagged, which features drove the score, and what the AI explanation
says. This is important for compliance, disputes, and model improvement.

---

## Key Results

All numbers are from the IEEE-CIS holdout set (118,108 transactions, 4,133 actual fraud).

### At the Optimised Threshold (0.53):

| Metric | Value |
|--------|-------|
| Fraud caught (True Positives) | 2,630 transactions |
| Fraud missed (False Negatives) | 1,503 transactions |
| Good customers wrongly blocked (False Positives) | 1,554 transactions |
| Precision | 62.9% (of blocked transactions, 63% are real fraud) |
| Recall | 63.6% (of all fraud, 64% is caught) |
| AUC-ROC | 0.948 (near-perfect ranking ability) |

### Financial Impact (Real Transaction Amounts):

| Item | Amount |
|------|--------|
| Fraud losses prevented (TP × 4.60 multiplier) | +$1,851,809 |
| Fraud losses incurred (FN × 4.60 multiplier) | −$1,088,211 |
| False positive revenue impact (FP × 1.63 multiplier) | −$687,477 |
| Human review cost (~59 cases × $16.33) | −$963 |
| **Net financial impact** | **+$75,165** |

### Compared to the Default High-Precision Setting (0.88):

| Setting | Net Impact |
|---------|-----------|
| Default threshold (0.88) | −$826,484 (loss) |
| Optimised threshold (0.53) | +$75,165 (profit) |
| **Improvement** | **+$901,649** |

The default setting catches only 36% of fraud with 95% precision — it almost never
makes a mistake, but it misses most of the fraud. The optimised setting catches 64%
of fraud, accepting more false positives, but the financial maths works out better
because the cost of missed fraud is much higher than the cost of a false positive.

---

## The Dashboard

ATLAS-X includes two dashboards:

### User Dashboard (http://localhost:5173)
For fraud operations teams. Shows:
- Live transaction feed with real-time decisions
- Fraud ring visualisation (force-directed graph)
- SHAP explainer — which features drove each decision
- AI explanation in plain English (powered by Qwen 2.5 locally, or GPT-4o-mini)
- Alert feed for high-risk transactions

### Backend Operations Dashboard (http://localhost:3000 / in-app)
For engineering and management. Four tabs:

**Ops Tab:**
System health, API latency, Kafka consumer lag, fraud ring summary, human review queue
(all FLAG decisions sorted by risk, with time waiting), and event audit trail.

**Ground Truth Tab:**
Confusion matrix, precision/recall/F1 by customer segment (VIP, Regular, New),
precision-recall curve, score distribution histogram, and threshold analysis table.

**Financial Impact Tab:**
Real dollar amounts from actual transaction data. Shows exactly how much money was
saved, lost, and spent on false positives and human review — at whatever threshold
is currently active.

**Threshold Optimizer Tab:**
Interactive chart showing net financial impact across all 91 thresholds. Drag the
slider to preview any threshold, see live P&L numbers, and click Apply to change
the threshold instantly — no system restart required.

---

## Customer Segments

ATLAS-X segments customers into three groups based on account tenure and transaction
recency:

| Segment | Threshold | Rationale |
|---------|-----------|-----------|
| VIP | 0.72 | Long-standing, high-value customers. Higher tolerance for risk to avoid frustrating loyal customers. |
| Regular | 0.88 | Standard customers. Balanced approach. |
| New Account | 0.82 | New accounts are higher risk. Slightly lower threshold than Regular. |

When the threshold optimizer is applied, it overrides all segments with a single
global threshold. Per-segment optimisation is possible but requires more data per
segment to be statistically reliable.

---

## Fraud Ring Detection

295 fraud rings were detected in the dataset. The largest rings:

| Ring Type | Example | Cards | Fraud Cards | Fraud Rate |
|-----------|---------|-------|-------------|------------|
| Email domain | anonymous.com | 2,680 | 860 | 32.1% |
| Email domain | hotmail.com | 1,865 | 782 | 41.9% |
| Device | desktop-Windows | 3,392 | 991 | 29.2% |

A "fraud ring" is a group of cards that share a device, email domain, or billing
address, where a significant fraction of those cards have committed fraud. When a
new transaction comes in from a card connected to a known fraud ring, ATLAS-X
elevates the risk score even if the model score alone is borderline.

---

## Human-in-the-Loop

Not every decision is fully automated. FLAG decisions go to a human review queue,
sorted by fraud probability (highest risk first). Each entry shows:
- Transaction ID and fraud probability
- Graph risk score
- Customer segment
- How long it has been waiting for review
- Plain-English reason for the flag

This ensures that borderline cases — where the model is uncertain but the graph
signal is strong — get human judgment before a final decision is made.

---

## AI Explanations

Every transaction can be explained in plain English. The system uses two AI models:

**Qwen 2.5 (free, runs locally):**
Default explanation engine. Runs on the same machine as the API, no external API
calls, no cost. Typical response time: 6–20 seconds.

**GPT-4o-mini (OpenAI):**
Available for comparison. Click "Compare with GPT-4o-mini" in the dashboard to see
both explanations side by side. Uses the OpenAI API (small cost per call).

Example explanation for a BLOCK decision (fraud_prob=0.977, graph_risk=0.516):
> "This transaction has been blocked due to a high fraud probability of 97.7%.
> The elevated graph risk score of 0.516 indicates that this transaction is connected
> to a suspicious network, and the card shares the same device as three confirmed
> fraudulent cards, significantly increasing the risk profile."

---

## Compliance and Audit

ATLAS-X is built with compliance in mind:

**Full audit trail:** Every prediction is stored in PostgreSQL with a complete event
history. You can query the full timeline of any transaction: when it was scored, what
the decision was, and all the data that drove it.

**SQL injection prevention:** All database queries are validated before execution —
SELECT-only, whitelisted tables, no injection patterns. This protects against both
external attacks and internal mistakes.

**Explainability:** Every decision can be explained at the feature level (SHAP values)
and in plain English (LLM). This is increasingly required by financial regulators who
mandate that automated credit/fraud decisions be explainable to customers.

**Human review queue:** FLAG decisions are never fully automated — they go to a human
analyst. This provides a check on the model and creates a paper trail for disputed
transactions.

---

## Scalability

The system is designed to scale horizontally:

- The FastAPI application is stateless (except for in-memory stats). Multiple instances
  can run behind a load balancer.
- The Kafka consumer can be scaled by adding more consumer instances in the same group.
- The XGBoost model is read-only after loading — multiple workers share it safely.
- PostgreSQL and Neo4j are the stateful components; they scale vertically or via
  managed cloud services (RDS, Neo4j Aura).

At current settings: 20–30 transactions/second sustained throughput, 27ms average
latency. The system has been tested at 200 transactions/second with the Kafka producer.

---

## Technology Choices (Plain English)

| Component | What It Does | Why This One |
|-----------|-------------|--------------|
| XGBoost | Fraud scoring model | Best accuracy on tabular data, fast, explainable |
| Neo4j | Fraud ring graph | Purpose-built for relationship queries |
| Apache Kafka | Transaction streaming | Industry standard, handles high throughput |
| FastAPI | REST API | Fast, modern Python, auto-generates documentation |
| PostgreSQL | Transaction storage | Reliable, supports pgvector for similarity search |
| pgvector | Find similar frauds | "Show me past frauds that look like this one" |
| Grafana | Operations monitoring | Standard tool, connects to Prometheus metrics |
| React | User dashboard | Fast, component-based, large ecosystem |
| Qwen 2.5 | AI explanations | Free, runs locally, no data leaves the system |

---

## Cost Assumptions (Sources)

All financial calculations use 2024 industry benchmarks:

| Cost Item | Value | Source |
|-----------|-------|--------|
| Fraud cost multiplier | ×4.60 per $1 of fraud | Risk Solutions (ramp.com, 2024) |
| False positive revenue impact | ×1.63 per $1 blocked | Aite-Novarica Group (greip.io, 2024) |
| Customer churn from false decline | 33% reduce/stop spending | Aite-Novarica Group (greip.io, 2024) |
| Fraud analyst salary | $35/hr ($66,149/yr) | Zippia, 2024 |
| Time per manual review | 20 minutes | DuckDuckGoose AI white paper, 2024 |
| Overhead multiplier | ×1.40 | Benefits + tools + management |
| Chargeback fee | $50 average | directpaynet.com, 2024 |

Transaction amounts used in financial calculations are real values from the IEEE-CIS
dataset (`TransactionAmt` column), not industry averages. The multipliers above are
applied to these real amounts.

---

## Author

Thayaananthan Kanagaraj
thayaa1903@gmail.com
