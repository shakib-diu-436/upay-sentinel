# System Design

## High-level flow

```text
Transaction Input
      ↓
Feature & Context Layer
      ↓
Transaction Risk Model
      +
Behavioral Anomaly Model
      +
(Advanced) Graph Risk Model
      ↓
Risk Engine
      ↓
Explainability Layer
      ↓
Customer / Analyst Alert
      ↓
Investigation Assistant
      ↓
Human Action
      ↓
Feedback Loop
```

## Architecture rules

- Keep data preparation separate from inference.
- Keep business rules separate from ML predictions.
- Make model outputs traceable and explainable.
- Keep sensitive decision logic out of free-form LLM prompts.
- High-impact actions remain subject to human review.
