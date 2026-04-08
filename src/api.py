from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.forecasting import ForecastingEngine
from src.shock_scenario import ShockScenario
from src.scenario_metrics import ScenarioMetrics
from src.policy_playbook import PolicyPlaybook
from src.sector_analysis import SectorAnalysis
from src.career_advisor import CareerAdvisor
from src.insight_generator import InsightGenerator
from src.story_generator import StoryGenerator
from src.model_validator import ModelValidator


app = FastAPI(title="Unemployment Scenario API")


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
    Shared helper to load and preprocess the India unemployment series.
    """
    loader = DataLoader("data/raw/india_unemployment.csv", "India")
    df = loader.load_clean_data()
    df = Preprocessor().preprocess(df)
    return df


# -------- Simulation Endpoint --------
@app.post("/simulate")
def simulate_scenario(request: ScenarioRequest):
    """
    Runs baseline + scenario simulation and returns results,
    including high-level scenario indices.
    """

    df = _load_prepared_series()

    # Baseline forecast
    baseline = ForecastingEngine(
        forecast_horizon=request.forecast_horizon
    ).forecast(df)

    # Scenario simulation
    scenario = ShockScenario(
        shock_intensity=request.shock_intensity,
        shock_duration=request.shock_duration,
        recovery_rate=request.recovery_rate,
    ).apply(baseline)

    # Metrics
    metrics = ScenarioMetrics.compute_delta(baseline, scenario)

    # Policy interpretation + indices
    policy_cfg = PolicyPlaybook.get_policy(request.policy_name)
    indices = ScenarioMetrics.compute_indices(
        baseline_df=baseline,
        scenario_df=scenario,
        policy_name=request.policy_name or "None",
        policy_cost_label=policy_cfg.get("relative_cost"),
    )
    
    # RQI
    rqi = ScenarioMetrics.compute_rqi(scenario, request.recovery_rate)
    indices.update(rqi)
    
    # Early Warning Indicator
    usi = indices.get("unemployment_stress_index", 0)
    rqi_label = rqi.get("rqi_label", "")
    
    if usi > 40 or rqi_label == "Poor Recovery":
        status = "🔴 High Risk"
    elif usi > 20 or rqi_label == "Fast but Fragile":
        status = "🟡 Watch"
    else:
        status = "🟢 Stable"
        
    indices["early_warning"] = status

    # Sector Impact Analysis (RSSI + Resilience)
    sector_impact = SectorAnalysis.analyze_sectors(
        scenario_df=scenario,
        shock_intensity=request.shock_intensity,
        recovery_rate=request.recovery_rate
    )

    # Career Advice (Rule-based AI)
    career_advice = CareerAdvisor.generate_advice(sector_impact)
    
    # AI Insights (Feature 7)
    # We need a name for the scenario, but the request doesn't pass "Scenario A" etc.
    # We can just use "Simulated Scenario" or infer from policy.
    scen_name = request.policy_name if request.policy_name and request.policy_name != "None" else "Shock Scenario"
    
    ai_insights = InsightGenerator.generate_scenario_insights(
        scenario_name=scen_name,
        indices=indices,
        sector_impact=sector_impact
    )
    
    # Story Mode (Feature 8)
    story = StoryGenerator.generate_story(scenario, baseline)

    return {
        "baseline": baseline.to_dict(orient="records"),
        "scenario": scenario.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "policy": policy_cfg,
        "indices": indices,
        "sector_impact": sector_impact.to_dict(orient="records"),
        "career_advice": career_advice,
        "ai_insights": ai_insights,
        "story": story,
    }


# -------- Backtesting / Model Trust Endpoint --------
@app.post("/backtest")
def backtest_model(request: BacktestRequest):
    """
    Performs a simple backtest on the most recent N years to
    illustrate how the trend-based forecasting behaves out of sample.
    """
    df = _load_prepared_series()

    test_years = max(1, min(request.test_years, 10))

    if len(df) <= test_years + 5:
        # Not enough history for a meaningful split; fall back to
        # a very small test window.
        test_years = min(3, max(1, len(df) - 3))

    train_df = df.iloc[:-test_years]
    test_df = df.iloc[-test_years:]

    engine = ForecastingEngine(forecast_horizon=test_years)
    forecast_df = engine.forecast(train_df)

    merged = test_df.merge(forecast_df, on="Year", how="inner")

    if merged.empty:
        return {
            "historical": [],
            "backtest": [],
            "mae": None,
            "mape": None,
        }

    errors = merged["Predicted_Unemployment"] - merged["Unemployment_Rate"]
    mae = float(errors.abs().mean())

    non_zero_actuals = merged["Unemployment_Rate"].replace(0, float("nan"))
    mape_series = (errors.abs() / non_zero_actuals) * 100.0
    mape = float(mape_series.mean())

    return {
        "historical": test_df.to_dict(orient="records"),
        "backtest": merged.to_dict(orient="records"),
        "mae": round(mae, 3),
        "mape": round(mape, 2) if mape == mape else None,
    }

# -------- Model Validation Endpoint --------
@app.get("/validate")
def validate_model():
    """
    Comprehensive model validation using multiple metrics:
    - MAE, MAPE, RMSE: Absolute error metrics
    - R²: Variance explained
    - Directional Accuracy: % correct predictions
    - Forecast Bias: Over/underestimation
    """
    df = _load_prepared_series()
    
    # Use last 60% as test, first 40% as train
    split_idx = int(len(df) * 0.4)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Forecast on test period
    forecast_horizon = len(test_df)
    engine = ForecastingEngine(forecast_horizon=forecast_horizon)
    forecast_df = engine.forecast(train_df)
    
    # Get validation report
    validation_report = ModelValidator.get_validation_report(test_df, forecast_df)
    
    return validation_report