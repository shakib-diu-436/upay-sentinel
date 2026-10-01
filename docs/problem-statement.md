# Upay Sentinel — Problem Statement & Project Definition

## 1. Project Name

Upay Sentinel

### Tagline
AI Financial Safety & Intelligence Layer

## 2. User

### Primary Users
- Digital wallet customers
- Risk and fraud monitoring teams
- Operations / investigation teams

### Secondary Stakeholders
- upay
- Customers whose transactions need additional protection

## 3. Problem

Digital financial transactions can become risky when unusual behavior, suspicious recipients,
abnormal transaction patterns, device changes, or unusual transaction activity occur together.

A transaction that looks normal based on a single signal may become suspicious when multiple
behavioral and contextual signals are considered together.

This creates two important challenges:

1. Customers may not easily understand whether a transaction is unusual or risky.
2. Operations and risk teams may need to investigate multiple signals manually when reviewing an alert.

Therefore, there is a need for an intelligent layer that can analyze transaction and behavioral signals together,
identify abnormal activity, explain the major reasons behind the alert, and support investigation.

## 4. Why AI?

The risk of a transaction may depend on multiple interacting signals rather than one fixed rule.

Example signals include:
- transaction amount
- recipient novelty
- transaction frequency
- transaction timing
- device changes
- location changes
- historical customer behavior

An AI/ML system can learn patterns from multiple features and identify unusual combinations of signals.

## 5. Solution

Upay Sentinel is an AI-powered financial safety and risk intelligence layer designed to analyze transaction
and behavioral signals and identify potentially suspicious activity.

The system will:
1. Analyze a transaction and its contextual signals.
2. Estimate a transaction risk score.
3. Compare current activity with historical behavioral patterns.
4. Identify important risk factors.
5. Explain why the transaction was considered unusual or risky.
6. Present an alert to the relevant user or operator.
7. Generate an evidence-based investigation summary for high-risk cases.
8. Support human decision-making rather than making high-impact financial decisions autonomously.

## 6. AI Role

### A. Transaction Risk Prediction
Possible approach: XGBoost / LightGBM

### B. Behavioral Anomaly Detection
Possible approach: Isolation Forest / Autoencoder / Clustering

### C. Explainability
Possible approach: SHAP / feature attribution / rule trace

### D. Investigation Assistance
Possible approach: LLM / RAG grounded in structured evidence

## 7. Expected Impact

### Customer-side impact
- Help customers recognize unusual transaction patterns.
- Provide understandable explanations for risk alerts.
- Support safer and more informed decisions.

### Operational impact
- Prioritize potentially high-risk cases.
- Reduce manual effort during initial investigation.
- Provide structured evidence and case summaries.

### Business impact
- Improve risk-monitoring efficiency.
- Support identification of abnormal transaction behavior.
- Provide a scalable AI-based risk intelligence layer for future validation.

## 8. Data Strategy

The hackathon prototype will use synthetic and/or self-generated data. No production customer data is required.

### Main data domains
- Customers
- Transactions
- Recipients
- Transaction timestamps
- Devices
- Locations
- Historical transaction behavior
- Transaction velocity
- Behavioral patterns

### Synthetic patterns
- Normal transaction behavior
- Abnormal behavioral deviations
- Suspicious/fraud-like patterns

No real personally identifiable information is used.

## 9. Validation

### ML metrics
- Precision
- Recall
- F1-score
- ROC-AUC
- False Positive Rate

### Baseline comparison
Compare the ML model against a simple deterministic rule-based baseline to demonstrate whether AI adds value.

### Operational metrics
- High-risk cases prioritized
- Suspicious cases detected
- Investigation time saved through structured summaries
- Coverage of explanations

## 10. Scale & Future Path

The prototype starts with synthetic data. Future validation could use appropriately governed, anonymized,
or aggregated real-world data if available.

Future modules:
- Account takeover intelligence
- Suspicious-network / money-mule detection
- Advanced behavioral profiling
- Real-time monitoring
- Investigation workflow integration
