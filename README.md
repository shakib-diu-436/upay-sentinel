# Upay Sentinel

## AI Financial Safety & Intelligence Layer

**Team:** AlphaCube (α³)  
**Institution:** Daffodil International University (DIU)  
**Project type:** AI Hackathon prototype — synthetic data only  
**Track:** Trust & Risk Intelligence  

> Upay Sentinel is an AI-powered financial safety and investigation layer for a digital wallet. It combines transaction risk scoring, behavioural anomaly detection, explainable evidence, account-takeover intelligence and suspicious-network analysis to help human risk analysts detect, understand and investigate potentially suspicious activity.

---

## 1. Project Overview

Digital-wallet risk is rarely explained by a single signal. A new recipient, changed device, changed location, unusual transaction amount, high velocity or unusual transaction time may be harmless individually, but combinations of these signals can indicate abnormal activity.

**Upay Sentinel** turns these signals into a structured analyst workflow:

**Detect → Explain → Investigate → Human Action**

The system produces:

1. A supervised transaction-risk score using XGBoost.
2. An unsupervised behavioural anomaly score using Isolation Forest.
3. A fused 0–100 Sentinel severity score.
4. Feature-level SHAP explanations for model-driven risk.
5. Rule-based investigation evidence.
6. Account-takeover intelligence.
7. Suspicious recipient / possible mule-network intelligence.
8. Persistent investigation cases in SQLite.
9. JWT-protected analyst APIs.
10. A React analyst dashboard for live investigation and case history.

The system is designed as **decision support**. It does not autonomously approve, deny or block consequential financial transactions.

---

## 2. Core Problem

Risk and operations teams may need to combine many contextual signals when reviewing suspicious wallet activity. Simple fixed rules can detect obvious patterns but are less useful for explaining model-driven risk, combining behavioural context and prioritising cases for investigation.

### Problem statement

> For digital-wallet risk and investigation teams, suspicious transaction activity can be difficult to identify and explain because risk depends on multiple behavioural and contextual signals. Upay Sentinel uses synthetic transaction history to detect abnormal activity, estimate composite risk, explain major risk drivers and support the first stage of human investigation.

---

## 3. Solution Architecture

```text
                Transaction Context
                         │
                         ▼
              Feature & Context Layer
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        XGBoost Risk          Isolation Forest
             │                 Behaviour Anomaly
             └───────────┬───────────┘
                         ▼
                    Risk Fusion
                         │
                         ▼
               Sentinel Risk Score
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        SHAP         Investigation      ATO
    Explanation       Evidence       Intelligence
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                Network / Mule Analysis
                         │
                         ▼
                Persistent Case Record
                         │
                         ▼
                    Human Review
```

The architecture separates data preparation, model inference, business/risk logic and investigation output so the core decision path remains inspectable and future integrations can replace individual components.

---

## 4. Implemented Features

### 4.1 Transaction Risk Scoring

- XGBoost binary classifier.
- Uses transaction, customer-history and contextual features.
- Validation-tuned classification threshold.
- Current saved model threshold: **0.52 (52%)**.

### 4.2 Behavioural Anomaly Detection

- Isolation Forest.
- Uses engineered behavioural-deviation features.
- Produces a 0–100 anomaly severity score.
- Hold-out evaluation is supported separately from supervised-model evaluation.
- The anomaly score is **not** presented as a calibrated fraud probability.

### 4.3 Risk Fusion

The transaction-risk and anomaly outputs are fused into a composite severity score:

```text
final = 0.70 × transaction_risk
      + 0.30 × anomaly
      + 0.15 × transaction_risk × anomaly
```

The resulting score is scaled to 0–100 and classified as:

| Score | Level | Prototype action |
|---:|---|---|
| 0–44.99 | LOW RISK | Continue under normal controls |
| 45–69.99 | REVIEW | Review context and supporting signals |
| 70–100 | HIGH RISK | Manual review and additional verification |

The fused score is a **composite severity score**, not a calibrated probability of fraud.

### 4.4 Explainability

- SHAP TreeExplainer for the XGBoost model.
- Top risk-driving features are returned with SHAP values.
- Each explanation states whether the feature increases or decreases model risk.
- Rule-based evidence is returned alongside model explanations.

### 4.5 Investigation Support

- Deterministic, auditable evidence rules.
- Plain-language evidence details.
- Structured investigation summary.
- Recommended human action.
- Case persistence in SQLite.
- Recent Cases can be reloaded after frontend refresh because cases are stored server-side.

### 4.6 Account-Takeover Intelligence

The project includes a transparent rule-based ATO module using signals such as:

- device changed
- location changed
- unusual transaction hour
- large deviation from usual transaction hour
- new recipient
- unusually high amount ratio
- high transaction velocity

The module returns an **ATO score, risk level, signals, recommended action and explanation**.

This is a rule-based intelligence layer, not a separate trained ML model.

### 4.7 Suspicious Network / Mule Detection

The project includes a graph-style network analysis module that examines customer–recipient relationships to identify suspicious recipient concentration and possible mule-network patterns.

The module returns:

- network risk score
- network classification
- focus recipient
- customer/recipient nodes
- transaction edges
- supporting signals
- human-readable network explanation

This network module is designed to support investigation; it does not autonomously label an account as a criminal/mule account.

### 4.8 Analyst Dashboard

- React + Vite interface.
- Transaction input form.
- Low-risk and high-risk demo presets.
- Risk score and risk level.
- Transaction information.
- Evidence cards.
- SHAP explanations.
- Investigation summary.
- ATO Intelligence panel.
- Suspicious Network / Mule Detection panel.
- Persistent Recent Cases panel.

---

## 5. AI / ML Components

| Component | Technology | Purpose |
|---|---|---|
| Transaction risk | XGBoost | Supervised suspicious-transaction classification |
| Behaviour anomaly | Isolation Forest | Unsupervised abnormal-behaviour detection |
| Explainability | SHAP | Feature-level risk attribution |
| Risk engine | Python | Score fusion and risk-level rules |
| ATO intelligence | Python rules | Account-takeover signal aggregation |
| Network intelligence | Python graph-style analysis | Recipient concentration and possible mule-network detection |
| Investigation support | Deterministic evidence rules | Auditable case explanation and summary |

No free-form LLM is required for the current investigation path.

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

The generator creates persistent customer profiles so behaviour is relative to each customer rather than globally. It tracks typical amounts, usual transaction hours, home locations, primary devices and recipient history.

Velocity features are computed from earlier events only, avoiding look-ahead leakage in those features. Synthetic suspicious scenarios inject combinations such as unusually large amounts, changed devices/locations, transaction bursts and unusual hours.

**Important limitation:** the synthetic suspicious label is generated from the same signal family used by the models. Offline results therefore show that the prototype works on its designed synthetic process; they do not establish production fraud-detection performance.

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

### Current operating-point results

The current evaluated XGBoost operating point uses a **0.52 classification threshold**. The current test-set recall is:

| Metric | Current result |
|---|---:|
| Recall | **66.51%** |
| False Negative Rate | **33.49%** |
| Classification threshold | **0.52** |

Run the project performance script to print the complete saved test metrics (Precision, Recall, F1, ROC-AUC, PR-AUC and confusion matrix) directly from `artifacts/risk_model.joblib`:

```bash
python ml/show_performance.py
```

The displayed results are **offline results on synthetic data** and should not be interpreted as production performance.

### Evaluation design updates

The evaluation workflow now includes separate hold-out evaluation for anomaly detection and evaluation of the fused risk score, rather than treating a single training/evaluation path as sufficient evidence for production readiness.

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
- SQLite
- PyJWT
- python-dotenv

### Frontend

- React
- Vite
- JavaScript / JSX
- CSS

### Data / Storage

- CSV-based synthetic transaction dataset
- Local model artifacts under `artifacts/`
- SQLite application database under `data/app/`

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

The current prototype uses **SQLite**, so no separate database server is required.

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

If backend dependencies are maintained separately, also install:

```bash
pip install -r backend/requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

---

## 11. Environment Variables & JWT Authentication

The backend uses JWT authentication for protected analyst APIs.

Create a local `.env` file in the project root:

```env
UPAY_JWT_SECRET=replace-with-a-long-random-secret
UPAY_JWT_EXPIRE_MINUTES=60
UPAY_ANALYST_USERNAME=analyst
UPAY_ANALYST_PASSWORD=replace-with-demo-password
```

**Never commit the real `.env` file or secret values to GitHub.**

A safe template is provided in `.env.example`:

```env
UPAY_JWT_SECRET=replace-with-a-secure-secret
UPAY_JWT_EXPIRE_MINUTES=60
UPAY_ANALYST_USERNAME=analyst
UPAY_ANALYST_PASSWORD=replace-with-demo-password
```

The frontend should obtain a short-lived JWT by calling the login endpoint and then send it using:

```text
Authorization: Bearer <access_token>
```

Protected API endpoints require a valid token.

---

## 12. Run the Project

Run these commands from the project root.

### Step 1 — Generate synthetic data

```bash
python ml/generator.py
```

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

### Step 4 — Show current model performance

```bash
python ml/show_performance.py
```

### Step 5 — Start FastAPI

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

### Step 6 — Start the React frontend

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

## 13. Authentication Quick Test

Start the backend and obtain a token with the analyst credentials configured in `.env`:

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=analyst&password=YOUR_PASSWORD"
```

The response contains:

```json
{
  "access_token": "...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "username": "analyst",
    "role": "risk_analyst"
  }
}
```

Use the token on protected endpoints:

```bash
TOKEN="YOUR_ACCESS_TOKEN"

curl -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/cases"
```

Without a token, protected endpoints should return an authentication error (`401`).

---

## 14. API Endpoints

### Public endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Health check |
| POST | `/api/v1/auth/login` | Analyst login and JWT issuance |
| GET | `/docs` | Swagger documentation |

### Protected endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/database/health` | SQLite database health/status |
| POST | `/api/v1/risk/analyze` | Risk score, level, recommendation and SHAP reasons |
| POST | `/api/v1/investigation/analyze` | Full investigation case |
| GET | `/api/v1/cases` | Recent persisted cases |
| GET | `/api/v1/cases/{case_id}` | Retrieve one investigation case |
| POST | `/api/v1/network/analyze` | Suspicious network / mule analysis |
| POST | `/api/v1/account-takeover/analyze` | Account-takeover analysis |
| POST | `/api/v1/intelligence/analyze` | Combined investigation + network + ATO intelligence |

All protected endpoints require:

```text
Authorization: Bearer <access_token>
```

### Example investigation request

After obtaining a JWT token:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/investigation/analyze" \
  -H "Authorization: Bearer $TOKEN" \
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

---

## 15. Database Persistence

The backend stores investigation data in a local SQLite database:

```text
data/app/upay_sentinel.db
```

The database contains:

- `transactions`
- `risk_assessments`
- `investigation_cases`

Case history is therefore stored server-side rather than only in browser state.

Check database status:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/database/health"
```

The database file is ignored by Git and should not be committed.

---

## 16. Testing / Verification

A practical end-to-end verification is:

1. Generate synthetic data.
2. Train XGBoost and Isolation Forest.
3. Start FastAPI.
4. Login through `/api/v1/auth/login` and obtain a JWT.
5. Open `/docs` and authorize with the bearer token.
6. Call `/api/v1/investigation/analyze` with a demo transaction.
7. Start the React frontend.
8. Analyse the **Low-Risk Example**.
9. Analyse the **High-Risk Example**.
10. Confirm the result shows:
   - transaction risk score
   - anomaly score
   - fused Sentinel score
   - risk level
   - evidence
   - SHAP reasons
   - investigation summary
   - recommended human action
   - ATO intelligence
   - suspicious-network intelligence
11. Refresh the frontend and confirm **Recent Cases** are still available from the database.
12. Run `python ml/show_performance.py` to verify saved model metrics.

Automated unit/integration test coverage is still a planned improvement.

---

## 17. Responsible AI & Safety

Upay Sentinel follows a human-in-the-loop approach.

### Privacy

Only synthetic/self-generated data is used in the hackathon prototype.

### Explainability

Important outputs include SHAP feature attributions and explicit rule-based evidence.

### Human oversight

The system recommends investigation and verification; it does not autonomously approve, deny or block consequential financial activity.

### Transparency

The fused Sentinel result is a composite severity score rather than a calibrated probability of fraud.

### Auditability

Investigation evidence and ATO/network signals are generated from explicit, inspectable logic so an analyst can trace why the system raised a case.

### Security

Protected analyst APIs use JWT bearer authentication. Secrets are kept in environment variables and excluded from version control.

---

## 18. Limitations

The current prototype has several known limitations:

- Synthetic labels are generated from the same signal family used by the models.
- The anomaly model and supervised model share some underlying behavioural signals.
- Fusion weights and score cut-offs are manually configured rather than calibrated on real analyst outcomes.
- The API still expects contextual transaction features from the caller; a production feature store/customer-history service is not yet present.
- SQLite is suitable for the prototype but is not a replacement for a production-scale operational data platform.
- Authentication is implemented, but production deployment would also need rate limiting, centralized secret management, stronger identity management and structured security logging.
- Model versioning, drift monitoring and real-time streaming ingestion are not yet implemented.
- ATO and network intelligence are transparent rule/analytics modules rather than separately trained predictive ML models.
- No automated test suite is currently included.
- Validation is synthetic and should not be interpreted as evidence of production fraud-detection performance.

---

## 19. Future Work

- Independent validation using appropriately governed real-world or independently generated data.
- Probability calibration and outcome-based tuning of fusion weights/cut-offs.
- More advanced graph analytics / Graph ML for mule-network discovery.
- Richer session, device-fingerprint and behavioural signals for account-takeover detection.
- Analyst feedback loop and controlled model retraining.
- Rate limiting and structured security/operational logging.
- Model versioning and drift monitoring.
- Feature-store integration and real-time event ingestion.
- Automated unit/integration testing and CI.
- Optional evidence-grounded RAG/LLM layer that only rephrases structured evidence and keeps the evidence list as the source of truth.

---

## 20. Repository Structure

```text
upay-sentinel/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── docs/
│   ├── problem-statement.md
│   └── system-design.md
├── data/
│   ├── synthetic/
│   └── processed/
│       └── app/
├── artifacts/
├── notebooks/
├── ml/
│   ├── generator.py
│   ├── train.py
│   ├── predict.py
│   ├── explain.py
│   ├── anomaly.py
│   ├── risk_engine.py
│   ├── show_performance.py
│   ├── network_analysis.py
│   └── account_takeover.py
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── auth.py
│       ├── schemas.py
│       ├── database.py
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

## 21. Live Deployment

**Current status:** Local prototype; no public deployment URL has been configured yet.

Before final submission, replace this section with the actual deployed URL if a live deployment is provided for judges.

```text
Live URL: https://<your-deployed-domain>
```

---

## 22. Competition / Disclosure Notes

This project is a hackathon prototype using synthetic data. External AI tools, frameworks, open-source libraries and development resources may be used where permitted by the competition rules.

The team is responsible for understanding the submitted implementation and should be able to explain the model choices, data-generation process, architecture, evaluation, security controls and responsible-AI decisions during judging.

No real upay production customer data is included in this repository.

---

## 23. Team

**AlphaCube (α³)**  
Daffodil International University

---

## License

This repository is intended for the AI DEV FEST 2026 hackathon prototype and demonstration.
