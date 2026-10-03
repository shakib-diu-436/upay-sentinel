# Upay Sentinel

## AI Financial Safety & Intelligence Layer

**Team:** AlphaCube (α³)  
**Institution:** Daffodil International University (DIU)  
**Project type:** AI Hackathon prototype — synthetic data only  
**Track:** Trust & Risk Intelligence  

> Upay Sentinel is a prototype AI risk-intelligence layer for a digital wallet. It combines transaction-level risk scoring, behavioural anomaly detection and explainable evidence to help human operators identify and investigate suspicious transaction activity.

---

## 1. Project Overview

Digital-wallet risk is rarely explained by a single signal. A large transaction, new recipient, changed device, unusual location or late-night transaction may be harmless by itself, but several signals appearing together can indicate abnormal activity.

**Upay Sentinel** addresses this problem by analysing multiple transaction and behavioural signals together and producing:

1. A supervised transaction-risk score.
2. An unsupervised behavioural anomaly score.
3. A fused 0–100 composite severity score.
4. An explainable set of major risk drivers using SHAP.
5. Structured evidence for investigation.
6. A human-review recommendation rather than an autonomous financial decision.

The prototype is designed primarily for risk/fraud monitoring and investigation workflows.

---

## 2. Core Problem

Risk and operations teams may need to manually combine multiple contextual signals when reviewing a suspicious transaction. A simple fixed-rule system can detect obvious patterns but may not capture more complex interactions or previously unseen behavioural deviations.

### Problem statement

> For digital-wallet risk and investigation teams, suspicious transaction activity can be difficult to identify and explain because risk depends on multiple behavioural and contextual signals. Upay Sentinel uses synthetic transaction history to detect abnormal activity, estimate composite risk, explain major risk drivers and support the first stage of human investigation.

---

## 3. Solution

### Detect → Explain → Investigate → Human Action

```text
Transaction Context
        │
        ▼
Feature & Context Layer
        │
        ├───────────────┐
        ▼               ▼
   XGBoost        Isolation Forest
 Transaction       Behavioural
   Risk              Anomaly
        │               │
        └───────┬───────┘
                ▼
           Risk Fusion
                │
                ▼
      Sentinel Severity Score
                │
        ┌───────┴────────┐
        ▼                ▼
      SHAP          Rule-based
   Explanation        Evidence
        │                │
        └───────┬────────┘
                ▼
       Investigation Case
                │
                ▼
          Human Review
```

The current prototype does **not** autonomously approve, deny or block consequential financial transactions.

---

## 4. Implemented Features

### Transaction Risk Scoring
- XGBoost binary classifier.
- Uses transaction, customer-history and contextual features.
- Validation-tuned classification threshold.

### Behavioural Anomaly Detection
- Isolation Forest.
- Measures deviation from behavioural patterns using engineered deviation features.
- Reported as a 0–100 anomaly severity score, not as a fraud probability.

### Risk Fusion
The two normalized scores are combined as:

```text
final = 0.70 × transaction_risk
      + 0.30 × anomaly
      + 0.15 × transaction_risk × anomaly
```

The final result is scaled to 0–100 and classified as:

| Score | Level | Prototype action |
|---:|---|---|
| 0–44.99 | LOW RISK | Continue under normal controls |
| 45–69.99 | REVIEW | Review context and supporting signals |
| 70–100 | HIGH RISK | Manual review and additional verification |

The fused score is a **composite severity score**, not a calibrated fraud probability.

### Explainability
- SHAP TreeExplainer.
- Returns the top five features by absolute SHAP magnitude.
- Shows whether each feature increases or decreases risk.

### Investigation Support
- Deterministic, auditable evidence rules.
- Plain-language evidence details.
- Structured investigation summary.
- Human-action recommendation.
- Explicit notice that the output supports, but does not replace, human decision-making.

### Analyst Dashboard
- React + Vite interface.
- Transaction input form.
- Low-risk and high-risk demo presets.
- Risk score and risk level.
- Transaction information.
- Evidence cards.
- SHAP explanations.
- Investigation summary.
- Session case list for comparing analysed transactions.

---

## 5. AI / ML Components

| Component | Technology | Purpose |
|---|---|---|
| Transaction risk | XGBoost | Supervised suspicious-transaction classification |
| Behavioural anomaly | Isolation Forest | Unsupervised abnormal-behaviour detection |
| Explainability | SHAP | Feature-level risk attribution |
| Risk engine | Python | Score fusion and risk-level rules |
| Investigation support | Deterministic evidence rules | Auditable case explanation and summary |

No free-form LLM is required for the current investigation workflow.

---

## 6. Data

The prototype uses **synthetic, self-generated data only**. No production upay customer data or real personally identifiable information is used.

### Dataset profile

- 20,000 transactions
- 3,000 customers
- 4,044 recipients
- 2,969 devices
- 500 locations
- 2026-07-01 to 2026-09-28
- 2,153 suspicious labels (10.77%)
- 17,847 normal labels (89.23%)
- 20 columns

### Synthetic-data design

The generator creates persistent customer profiles so that behaviour is relative to each customer rather than globally. It tracks typical amounts, usual transaction hours, home locations, primary devices and recipient history.

Velocity features are computed from earlier events only, avoiding look-ahead leakage in those features. Synthetic suspicious scenarios inject combinations such as unusually large amounts, changed devices/locations, transaction bursts and unusual hours.

**Important limitation:** the synthetic suspicious label is generated from the same signal family used by the models. Therefore, offline results demonstrate that the prototype works on its designed synthetic process; they do not establish production fraud-detection performance.

---

## 7. Model Training & Evaluation

### XGBoost setup

- Split: 60% train / 20% validation / 20% test
- Stratified split
- `random_state = 42`
- Class imbalance handled with `scale_pos_weight`
- 350 trees
- Max depth 5
- Learning rate 0.05
- Subsample 0.85
- Column sampling 0.85
- Classification threshold tuned on validation data using F1

### Current offline test results

Test set: **4,000 synthetic transactions**

| Metric | Result |
|---|---:|
| Accuracy | 92.20% |
| Precision | 72.52% |
| Recall | 44.19% |
| F1 Score | 54.91% |
| ROC-AUC | 84.68% |
| PR-AUC | 61.76% |

### Confusion matrix

```text
                 Predicted
              Normal  Suspicious
Actual Normal   3498      72
Actual Susp.     240     190
```

These results are **offline results on synthetic data** and should not be interpreted as production performance.

---

## 8. Technology Stack

### Machine Learning
- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Joblib

### Backend
- FastAPI
- Pydantic
- Uvicorn

### Frontend
- React
- Vite
- JavaScript / JSX
- CSS

### Data
- CSV-based synthetic transaction dataset
- Local prototype model artifacts

---

## 9. Requirements

### Software

- Python 3.10+ recommended
- `pip`
- Node.js and npm
- Git

### Recommended environment

- Linux/macOS/Windows with Python and Node.js available
- VS Code or another code editor

No external database is required for the current prototype.

---

## 10. Installation & Setup

Clone the public repository:

```bash
git clone git@github.com:shakib-diu-436/upay-sentinel.git
cd upay-sentinel
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

---

## 11. Environment Variables

The current local prototype does not require mandatory environment variables or API secrets.

Do **not** commit API keys, passwords, access tokens or other secrets to GitHub.

For a future production deployment, secrets should be stored outside the repository using the deployment platform's secret-management mechanism.

---

## 12. Run the Project

Run these commands from the project root.

### Step 1 — Generate synthetic data

```bash
python ml/generator.py
```

This generates the synthetic transaction dataset.

### Step 2 — Train the XGBoost model

```bash
python ml/train.py
```

This creates the supervised model artifact under `artifacts/`.

### Step 3 — Train the Isolation Forest

```bash
python ml/anomaly.py
```

This creates the anomaly-model artifact under `artifacts/`.

### Step 4 — Start FastAPI

```bash
uvicorn backend.app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

### Step 5 — Start the React frontend

Open a second terminal:

```bash
cd frontend
npm run dev
```

Frontend:

```text
http://127.0.0.1:5173
```

The backend must be running before analysing a transaction from the frontend.

---

## 13. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Health check |
| POST | `/api/v1/risk/analyze` | Risk score, level, recommendation and SHAP reasons |
| POST | `/api/v1/investigation/analyze` | Full investigation case with evidence and human-action guidance |

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/investigation/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX-DEMO-001",
    "customer_id": "C-DEMO-001",
    "recipient_id": "R-DEMO-001",
    "timestamp": "2026-10-03T01:20:00",
    "amount": 8000,
    "recipient_new": 1,
    "device_changed": 1,
    "location_changed": 1,
    "transactions_last_1h": 4,
    "transactions_last_24h": 8,
    "avg_amount_30d": 900,
    "usual_transaction_hour": 14,
    "hour": 1,
    "account_age_days": 400
  }'
```

You can also use the interactive Swagger UI at `/docs`.

---

## 14. Testing / Verification

A quick end-to-end verification is:

1. Generate the synthetic data.
2. Train both models.
3. Start FastAPI.
4. Open `/docs` and call `/api/v1/investigation/analyze`.
5. Start the React frontend.
6. Load the **Low-Risk Example** and analyse it.
7. Load the **High-Risk Example** and analyse it.
8. Confirm that the result contains:
   - transaction risk score
   - anomaly score
   - fused Sentinel score
   - risk level
   - evidence
   - SHAP model reasons
   - investigation summary
   - recommended human action

The current UI case list is session-based; persistent database-backed case storage is a future engineering step.

---

## 15. Responsible AI & Safety

Upay Sentinel follows a human-in-the-loop approach.

### Privacy
Only synthetic/self-generated data is used in the hackathon prototype.

### Explainability
Important outputs include SHAP feature attributions and rule-based evidence.

### Human oversight
The system recommends investigation and verification; it does not autonomously approve, deny or block consequential financial activity.

### Transparency
The fused Sentinel result is explicitly described as a composite severity score rather than a calibrated fraud probability.

### Auditability
Investigation evidence is generated from explicit, inspectable rules instead of hidden decision logic inside a free-form prompt.

---

## 16. Limitations

The current prototype has several known limitations:

- Synthetic labels are generated from the same signal family used by the models.
- Isolation Forest is demonstrated without an independent hold-out evaluation.
- The anomaly and supervised models share some underlying behavioural signals.
- Fusion weights and score cut-offs are manually set rather than calibrated using real outcomes.
- The API expects contextual features from the caller; there is no production feature store or customer-history service.
- The prototype does not yet include authentication, rate limiting, structured production logging, model-drift monitoring or real-time streaming ingestion.
- The investigation assistant is deterministic and evidence-based rather than a full RAG/LLM system.

These limitations are intentional and define the next validation and productionization steps.

---

## 17. Future Work

- Independent hold-out evaluation for anomaly detection.
- Probability calibration and outcome-based tuning of fusion weights/cut-offs.
- Graph analytics for suspicious recipient clusters and possible money-mule networks.
- Account-takeover intelligence using richer session and device signals.
- Analyst feedback loop and controlled model retraining.
- Authentication, rate limiting, structured logging and model versioning.
- Feature-store and real-time event ingestion.
- Controlled validation using appropriately governed anonymised or aggregated real-world data.
- Optional evidence-grounded RAG/LLM layer that only rephrases structured evidence.

---

## 18. Repository Structure

```text
upay-sentinel/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── problem-statement.md
│   └── system-design.md
├── data/
│   ├── synthetic/
│   └── processed/
├── ml/
│   ├── generator.py
│   ├── train.py
│   ├── predict.py
│   ├── explain.py
│   ├── anomaly.py
│   └── risk_engine.py
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       └── services/
│           ├── __init__.py
│           ├── model_service.py
│           ├── risk_service.py
│           └── investigation_service.py
└── frontend/
    ├── package.json
    ├── index.html
    └── src/
        ├── App.jsx
        ├── main.jsx
        └── index.css
```

---

## 19. Live Deployment

**Current status:** Local prototype; no public deployment URL has been configured yet.

Before final submission, replace this section with the actual deployed URL if a live deployment is provided for judges.

Example:

```text
Live URL: https://<your-deployed-domain>
```

---

## 20. Competition / Disclosure Notes

This project is a hackathon prototype using synthetic data. External AI tools, frameworks and open-source libraries are used as development resources where permitted by the competition rules.

The team is responsible for understanding the submitted implementation and should be able to explain the model choices, data-generation process, architecture, evaluation and responsible-AI decisions during judging.

No real upay production customer data is included in this repository.

---

## 21. Team

**AlphaCube (α³)**  
Daffodil International University

---

## License

This repository is intended for the AI DEV FEST 2026 hackathon prototype and demonstration.
