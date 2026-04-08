from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.live_data import fetch_world_bank, get_data_source_label
from src.preprocessing import Preprocessor
from src.forecasting import ForecastingEngine
from src.shock_scenario import ShockScenario
from src.scenario_metrics import ScenarioMetrics
from src.policy_playbook import PolicyPlaybook
from src.sector_analysis import SectorAnalysis
from src.career_advisor import CareerAdvisor
from src.llm_insights import generate_insights
from src.story_generator import StoryGenerator
from src.model_validator import ModelValidator
from src.historical_events import get_all_events


app = FastAPI(title="Unemployment Intelligence Platform API v2")


# -------- Request Schemas --------
class ScenarioRequest(BaseModel):
    shock_intensity: float
    shock_duration: int
    recovery_rate: float
    forecast_horizon: int = 6
    policy_name: Optional[str] = None


class BacktestRequest(BaseModel):
    test_years: int = 5


def _load_prepared_series():
    """
    Loads India unemployment series.
    Tries World Bank Live API first, falls back to local CSV automatically.
    """
    df = fetch_world_bank(country="India")
    df = Preprocessor().preprocess(df)
    return df


# -------- Simulation Endpoint --------
@app.post("/simulate")
def simulate_scenario(request: ScenarioRequest):
    df = _load_prepared_series()

    # Baseline forecast with confidence bands
    engine = ForecastingEngine(forecast_horizon=request.forecast_horizon)
    baseline = engine.forecast(df)
    baseline_conf = engine.forecast_with_confidence(df)

    # Scenario simulation (shock now hits immediately — fixed logic)
    scenario = ShockScenario(
        shock_intensity=request.shock_intensity,
        shock_duration=request.shock_duration,
        recovery_rate=request.recovery_rate,
    ).apply(baseline)

    # Metrics
    metrics = ScenarioMetrics.compute_delta(baseline, scenario)

    # Policy + Indices
    policy_cfg = PolicyPlaybook.get_policy(request.policy_name)
    indices = ScenarioMetrics.compute_indices(
        baseline_df=baseline,
        scenario_df=scenario,
        policy_name=request.policy_name or "None",
    )

    # Recovery Quality Index
    rqi = ScenarioMetrics.compute_rqi(scenario, request.recovery_rate)
    indices.update(rqi)

    # Early Warning
    usi = indices.get("unemployment_stress_index", 0)
    rqi_label = rqi.get("rqi_label", "")
    if usi > 40 or rqi_label == "Poor Recovery":
        status = "🔴 High Risk"
    elif usi > 20 or rqi_label == "Fast but Fragile":
        status = "🟡 Watch"
    else:
        status = "🟢 Stable"
    indices["early_warning"] = status

    # Sector Analysis (calibrated to India's historical peak)
    sector_impact = SectorAnalysis.analyze_sectors(
        scenario_df=scenario,
        shock_intensity=request.shock_intensity,
        recovery_rate=request.recovery_rate,
    )

    # Career Advice with dynamic thresholds
    career_advice = CareerAdvisor.generate_advice(
        sector_impact, shock_intensity=request.shock_intensity
    )

    # AI Insights (LLM or rule-based fallback)
    scen_name = (
        request.policy_name
        if request.policy_name and request.policy_name != "None"
        else "Shock Scenario"
    )
    ai_insights = generate_insights(
        scenario_name=scen_name,
        indices=indices,
        sector_impact=sector_impact,
    )

    # Story timeline
    story = StoryGenerator.generate_story(scenario, baseline)

    return {
        "baseline": baseline.to_dict(orient="records"),
        "baseline_confidence": baseline_conf.to_dict(orient="records"),
        "scenario": scenario.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "policy": policy_cfg,
        "indices": indices,
        "sector_impact": sector_impact.to_dict(orient="records"),
        "career_advice": career_advice,
        "ai_insights": ai_insights,
        "story": story,
        "data_source": get_data_source_label("India"),
    }


# -------- Backtesting Endpoint --------
@app.post("/backtest")
def backtest_model(request: BacktestRequest):
    df = _load_prepared_series()

    test_years = max(1, min(request.test_years, 10))
    if len(df) <= test_years + 5:
        test_years = min(3, max(1, len(df) - 3))

    train_df = df.iloc[:-test_years]
    test_df = df.iloc[-test_years:]

    engine = ForecastingEngine(forecast_horizon=test_years)
    forecast_df = engine.forecast(train_df)
    merged = test_df.merge(forecast_df, on="Year", how="inner")

    if merged.empty:
        return {"historical": [], "backtest": [], "mae": None, "mape": None}

    errors = merged["Predicted_Unemployment"] - merged["Unemployment_Rate"]
    mae = float(errors.abs().mean())
    non_zero = merged["Unemployment_Rate"].replace(0, float("nan"))
    mape = float((errors.abs() / non_zero * 100).mean())

    return {
        "historical": test_df.to_dict(orient="records"),
        "backtest": merged.to_dict(orient="records"),
        "mae": round(mae, 3),
        "mape": round(mape, 2) if mape == mape else None,
    }


# -------- Validation Endpoint --------
@app.get("/validate")
def validate_model():
    df = _load_prepared_series()
    split_idx = int(len(df) * 0.4)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    engine = ForecastingEngine(forecast_horizon=len(test_df))
    forecast_df = engine.forecast(train_df)
    return ModelValidator.get_validation_report(test_df, forecast_df)


# -------- Historical Events Endpoint --------
@app.get("/events")
def get_historical_events():
    return {"events": get_all_events()}


# -------- Data Source Status --------
@app.get("/data-status")
def data_status():
    label = get_data_source_label("India")
    return {"source": label, "country": "India"}
