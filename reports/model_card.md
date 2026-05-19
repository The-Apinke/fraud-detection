# Model Card — Fraud Detection Ensemble

## Model Details

| Field | Value |
|---|---|
| Version | 1.0.0 |
| Architecture | LightGBM + MLP → Logistic Regression stacked ensemble |
| Training date | TBD (fill after training) |
| Dataset | IEEE-CIS Fraud Detection (Kaggle) — 590k transactions |
| Framework | LightGBM 4.5, PyTorch, scikit-learn |

## Intended Use

**Primary use case:** Real-time fraud scoring for payment transactions, returning a probability score and a three-tier verdict (APPROVE / REVIEW / BLOCK).

**Intended users:** Risk analysts, fraud operations teams, payment platforms.

**Out-of-scope uses:** This model was trained on e-commerce transaction data. It should not be used for:
- Insurance fraud
- Identity fraud (no account-level features)
- Any domain substantially different from online payments

## Training Data

- **Source:** IEEE-CIS Fraud Detection dataset (Vesta Corporation)
- **Size:** ~590k transactions across 30 days
- **Fraud rate:** ~3.5% (severe class imbalance)
- **Split strategy:** Time-based (train on earlier transactions, test on later ones) to prevent data leakage and simulate real-world deployment where the model always predicts future events from past patterns.
  - Train: 70% (earliest transactions)
  - Validation: 15%
  - Test: 15% (most recent — locked until final evaluation)

## Performance

*Fill these values after running `05_evaluation.ipynb`:*

| Metric | Value |
|---|---|
| ROC-AUC (test) | — |
| PR-AUC (test) | — |
| Precision @ block threshold | — |
| Recall @ block threshold | — |
| False Positive Rate @ block threshold | — |

**Operating thresholds:**
- **BLOCK (≥0.70):** High confidence fraud. Transaction declined.
- **REVIEW (0.35–0.70):** Uncertain. Routed to human analyst or step-up authentication.
- **APPROVE (<0.35):** Low fraud risk. Transaction approved.

## Business Cost Framework

The threshold was selected by minimising a cost function, not by maximising accuracy.

| Error type | Business cost |
|---|---|
| False positive (blocking a legitimate transaction) | ~$10 per incident (customer service cost + churn risk) |
| False negative (approving a fraudulent transaction) | ~$150 per incident (fraud loss + chargeback cost) |

**Why false positives matter at scale:** At 10 million daily transactions with a 1% false positive rate, 96,500 legitimate customers are blocked every day. Each one represents a declined card, a potential customer service call, and a risk of permanent churn. This is why the model is tuned for high precision at the BLOCK threshold, accepting a lower recall.

## Limitations and Failure Modes

1. **Concept drift:** Fraud patterns evolve. Model performance will degrade over time as fraudsters adapt to detection patterns. Retraining on recent data every 30–90 days is recommended.

2. **New fraud patterns:** The model cannot detect novel fraud types it has never seen. A rule-based system should run alongside the model to catch emerging patterns immediately.

3. **Sparse features:** Many V-columns in the dataset have >50% missing values. The model handles this via -999 imputation, but predictions on transactions with many missing features are less reliable.

4. **Cold-start problem:** New cardholders with no transaction history produce less reliable velocity and behavioural deviation features. Scores for new cards should be treated with more caution.

5. **Geographic bias:** The training data represents a specific geographic and merchant mix. Performance may be lower on transaction types underrepresented in the IEEE-CIS dataset.

## Ethical Considerations

- **Disparate impact:** Fraud models can inadvertently encode demographic biases if certain groups are overrepresented in historical fraud data. This model has not been audited for demographic disparate impact and should not be deployed in contexts where such auditing is legally required without additional analysis.

- **Explainability:** Every prediction includes SHAP-based explanations. Human reviewers should always be able to understand why a transaction was flagged before permanent action is taken.

- **Right to appeal:** Customers whose legitimate transactions are blocked should have a clear, fast path to manual review. The REVIEW tier exists specifically to route uncertain cases to human judgment rather than automatic blocking.

## How to Update This Card

After running `05_evaluation.ipynb`, copy the values from `final_metrics.json` into the Performance table above. Also update `api/main.py` → `MODEL_METRICS` dict so the API serves live metrics.
