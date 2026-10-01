# Upay Sentinel

**AI Financial Safety & Intelligence Layer**

Upay Sentinel is a hackathon prototype for detecting suspicious digital-wallet transaction behavior, explaining the major risk signals, and assisting human operators with evidence-based investigation summaries.

## Current MVP
1. Synthetic transaction dataset
2. Transaction risk prediction
3. Behavioral anomaly detection (next phase)
4. Explainable risk factors
5. Investigation assistant (next phase)
6. Suspicious-network graph analysis (advanced phase)

## Project flow
Transaction → Feature Engineering → Risk Model → Risk Score → Explanation → Alert → Investigation → Human Action → Feedback

## Folder structure

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
├── notebooks/
│   └── model_experiment.ipynb
├── ml/
│   ├── generator.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── backend/
│   ├── requirements.txt
│   └── app/
│       └── main.py
└── frontend/
    └── README.md
```

## Run the first phase

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Ubuntu/Linux
source .venv/bin/activate

pip install -r requirements.txt
python ml/generator.py
python ml/train.py
python ml/predict.py
```

The generated data and model artifacts are stored under `data/` and `artifacts/`.
# upay-sentinel
