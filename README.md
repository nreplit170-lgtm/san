w# Unemployment Trend Analysis

Scenario-based unemployment forecasting with shock simulation, policy playbook, and interactive analytics.

## Problem Statement

Traditional unemployment forecasting often assumes stable conditions. This platform supports **what-if** analysis:

- What if a severe economic shock occurs?
- How quickly does unemployment recover?
- How do policy interventions affect outcomes?

**Features:** Baseline forecasting (trend + mean reversion), shock/recovery scenarios, scenario comparison, policy levers, sector analysis, career insights, and an API + Streamlit UI.

## Tech Stack

| Layer      | Technologies                          |
|-----------|---------------------------------------|
| Backend   | Python, FastAPI, Pandas, NumPy       |
| Frontend  | Streamlit, Plotly                    |
| Data      | World Bank Open Data, local CSV      |

## Project Structure

```
Unemployment/
├── app.py                 # Streamlit UI (entry point for frontend)
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/                   # Core package (production code only)
│   ├── __init__.py
│   ├── api.py             # FastAPI backend
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── event_detection.py # Regime labels: Shock / Recovery / Stable
│   ├── forecasting.py    # Trend + mean-reversion engine
│   ├── shock_scenario.py # Shock overlay with decay
│   ├── scenario_engine.py
│   ├── scenario_metrics.py
│   ├── risk_engine.py
│   ├── policy_playbook.py
│   ├── sector_analysis.py
│   ├── career_advisor.py
│   ├── insight_generator.py
│   ├── insights.py
│   ├── story_generator.py
│   └── model_validator.py
│
├── data/
│   ├── raw/               # Input CSVs (e.g. india_unemployment.csv)
│   └── API_*/             # World Bank indicator data (optional)
│
├── docs/
│   └── RESEARCH_PAPERS_REFERENCES.md
│
└── tests/                 # Add pytest tests here
    └── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend (FastAPI)

```bash
uvicorn src.api:app --reload
```

Backend: **http://127.0.0.1:8000**  
API docs: **http://127.0.0.1:8000/docs**

### 3. Start the frontend (Streamlit)

In a new terminal:

```bash
streamlit run app.py
```

Frontend: **http://localhost:8501**

## How the app works

1. User selects or configures two economic scenarios (shock intensity, duration, recovery rate, policy).
2. Clicks **Compare Scenarios**.
3. App shows baseline forecast, Scenario A & B, comparison chart, and metrics (via backend API).

## Data

- **Indicator:** Unemployment, total (% of labor force) — World Bank `SL.UEM.TOTL.ZS`.
- Local series (e.g. `data/raw/india_unemployment.csv`) for India or other countries.

## Assumptions & limitations

- Baseline uses **trend + mean reversion** (interpretable, not ML).
- Shocks are parametric overlays with decay; no causal GDP/inflation in the model.
- Suited for **decision support and scenario analysis**, not real-time prediction.

## Research references

See **docs/RESEARCH_PAPERS_REFERENCES.md** for literature (forecasting, shock scenarios, youth unemployment, policy).

## Author

Bhushan Nanavare — Full-Stack & Analytics Developer
