# Unemployment Intelligence Platform

A Streamlit + FastAPI application for scenario-based unemployment forecasting, shock simulation, and policy analysis.

## Architecture

Two services run simultaneously:
- **FastAPI backend** (`src/api.py`) on port 8000 — handles simulation, forecasting, and validation endpoints
- **Streamlit frontend** (`app.py`) on port 5000 — multi-page dashboard UI

## Project Structure

```
app.py                  # Streamlit home page (entry point)
pages/                  # Streamlit multipage pages (1–10)
src/
  api.py                # FastAPI routes: /simulate, /backtest, /validate
  data_loader.py        # World Bank CSV ingestion
  preprocessing.py      # Smoothing and cleaning
  forecasting.py        # Trend + mean-reversion engine
  shock_scenario.py     # Economic shock overlay
  scenario_metrics.py   # USI, peak delta, RQI indices
  policy_playbook.py    # Policy catalogue
  sector_analysis.py    # Sector stress / resilience scoring
  career_advisor.py     # Rule-based career skill advice
  insight_generator.py  # Narrative text insights
  story_generator.py    # Timeline story from scenario data
  model_validator.py    # MAE, MAPE, RMSE, R², directional accuracy
  skill_obsolescence.py # Emerging vs declining skill detection
  ui_helpers.py         # Shared CSS, KPI cards, Plotly dark layout
  job_market_pulse.py   # Job posting trend analysis
  job_risk_model.py     # ML-based job risk predictor
  geo_career_advisor.py # Location-based career insights
data/
  raw/india_unemployment.csv     # Primary dataset
  market_pulse/job_postings_sample.csv
  geo/india_city_reference.csv
```

## Workflows

- `FastAPI Backend` — `uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload`
- `Start application` — `streamlit run app.py --server.port 5000 ...`

## Tech Stack

| Layer     | Technologies                              |
|-----------|-------------------------------------------|
| Backend   | Python 3.11, FastAPI, Pandas, NumPy, SciPy, scikit-learn |
| Frontend  | Streamlit, Plotly, Folium, streamlit-folium |
| Data      | World Bank Open Data, local CSV files    |

## Key Features

1. Scenario Simulator — compare two economic shock scenarios side by side
2. Sector Analysis — RSSI heatmap and resilience scoring for 5 sectors
3. Career Lab — skill recommendations from sector stress analysis
4. AI Insights ✅ — narrative text, policy comparison, shock sensitivity table, export report
5. Model Validation ✅ — backtest metrics, R² gauge, residuals chart, error distribution, year-by-year table, export report
6. Job Risk Predictor ✅ — feature contribution chart, industry comparison, what-if skill upgrade diff, export report
7. Job Market Pulse ✅ — skill momentum badges, location demand chart, skill gap analyzer, export
8. Geo Career Advisor ✅ — Folium map, city volume+salary chart, LQ bar chart, coloured relocation ranking, all-tier risk comparison, export
9. Skill Obsolescence Detector ✅ — statistical trend detection (scipy linregress), KPI summary, trend scatter, multi-skill timeline, personal skills-at-risk checker, export

## API Health Fix
- `API_BASE_URL` changed to `http://127.0.0.1:8000` (avoids DNS delay, prevents "API Offline" on startup)
- Health check TTL lowered to 10 s with 2-attempt retry via `/data-status` endpoint

## Data Modes

The platform supports two parallel data modes:

| Mode | Description | Pages |
|------|-------------|-------|
| **🌐 Real Data** | Live World Bank Open Data API (no key required) | Overview (evidence-based), Sector → Live tab, Market Pulse → Live tab, Geo Career |
| **🧪 Simulation** | Parametric shock equations seeded from India's historical baseline | Simulator, Career Lab, AI Insights, Sector → Scenario tab |

## Free AI Integration

Multi-provider LLM chain in `src/llm_insights.py`:
1. **Groq (LLaMA 3.1 · Free)** — `GROQ_API_KEY` set as Replit secret
2. Gemini (free) — `GEMINI_API_KEY` optional
3. OpenAI (paid) — `OPENAI_API_KEY` optional
4. Rule-based fallback — always available

## Development Phases
- Phase 1 (done): Features 1–3 (Simulator, Sector Analysis, Career Lab)
- Phase 2 (done): Features 4–5 (AI Insights, Model Validation) + API health fix
- Phase 3 (done): Feature 6 (Job Risk Predictor)
- Phase 4 (done): Feature 7 (Job Market Pulse)
- Phase 5 (done): Feature 8 (Geo Career Advisor)
- Phase 6 (done): Feature 9 (Skill Obsolescence) — ALL 9 FEATURES COMPLETE
- Phase 7 (done): Remaining polish — dynamic status bar, simulation mode banners on pages 2/4/5, real data backtest badge on page 6, "Two Data Modes" homepage section, 56× use_container_width deprecation fixes, Arrow mixed-type table bug fix
