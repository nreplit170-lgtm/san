"""
llm_insights.py
LLM-powered economic narrative generator.

Tries OpenAI GPT-4o-mini first (if OPENAI_API_KEY is set).
Falls back to the rule-based InsightGenerator automatically — zero downtime.
"""
import os
import json
from src.insight_generator import InsightGenerator
import pandas as pd


def _build_prompt(scenario_name: str, indices: dict, sector_impact_df: pd.DataFrame) -> str:
    sectors = sector_impact_df[["Sector", "Stress_Score", "Resilience_Score"]].to_dict(orient="records")
    sector_str = "; ".join(
        f"{s['Sector']} (stress={s['Stress_Score']:.1f}, resilience={s['Resilience_Score']:.1f})"
        for s in sectors
    )
    return f"""You are a senior economist analysing Indian labor market data.

Scenario: {scenario_name}
Unemployment Stress Index (USI): {indices.get('unemployment_stress_index', 'N/A')}
Peak deviation from baseline: {indices.get('peak_delta', 'N/A')} percentage points
Years above baseline: {indices.get('years_above_baseline', 'N/A')}
Recovery Quality: {indices.get('rqi_label', 'N/A')}
Early Warning Status: {indices.get('early_warning', 'N/A')}
Policy in place: {indices.get('policy_cushion_score', 0)} cushion score
Sector impacts: {sector_str}

Write a concise, professional economic analysis (3 short paragraphs):
1. Macro impact: What does this mean for India's labor market overall?
2. Sector breakdown: Which sectors are most/least exposed and why?
3. Policy outlook: What interventions would be most effective and why?

Be specific, data-driven, and avoid generic statements. Write for a policymaker audience."""


def generate_insights(
    scenario_name: str,
    indices: dict,
    sector_impact: pd.DataFrame,
) -> dict:
    """
    Attempts LLM generation, falls back to rule-based on any failure.
    Returns dict with keys: summary, macro_insight, sector_insight, recovery_insight, source.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if api_key:
        try:
            import requests as _req
            prompt = _build_prompt(scenario_name, indices, sector_impact)
            response = _req.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 600,
                    "temperature": 0.4,
                },
                timeout=20,
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                return {
                    "summary": content,
                    "macro_insight": paragraphs[0] if len(paragraphs) > 0 else content,
                    "sector_insight": paragraphs[1] if len(paragraphs) > 1 else "",
                    "recovery_insight": paragraphs[2] if len(paragraphs) > 2 else "",
                    "source": "🤖 GPT-4o-mini",
                }
        except Exception:
            pass

    # Rule-based fallback
    result = InsightGenerator.generate_scenario_insights(scenario_name, indices, sector_impact)
    result["source"] = "📐 Rule-based (AI key not set)"
    return result
