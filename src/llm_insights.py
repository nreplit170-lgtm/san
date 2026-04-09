"""
llm_insights.py
LLM-powered economic narrative generator.

Provider priority (all failures fall through gracefully):
  1. Groq      — free tier, no credit card needed (GROQ_API_KEY)
                 14,400 req/day free · model: llama-3.1-8b-instant
                 Get key: https://console.groq.com  (sign-up is free)
  2. Gemini    — free tier with a Google account (GEMINI_API_KEY)
                 15 RPM / 1M tokens/day free · model: gemini-1.5-flash
                 Get key: https://aistudio.google.com/app/apikey
  3. OpenAI    — paid, optional premium (OPENAI_API_KEY)
                 model: gpt-4o-mini
  4. Rule-based — always works, zero setup, zero cost.
"""
import os
import requests as _req
import pandas as pd
from src.insight_generator import InsightGenerator


# ── shared prompt builder ─────────────────────────────────────────────────────

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


def _parse_paragraphs(content: str) -> tuple[str, str, str]:
    """Split LLM response into 3 insight paragraphs."""
    paragraphs = [p.strip() for p in content.strip().split("\n\n") if p.strip()]
    macro    = paragraphs[0] if len(paragraphs) > 0 else content
    sector   = paragraphs[1] if len(paragraphs) > 1 else ""
    recovery = paragraphs[2] if len(paragraphs) > 2 else ""
    return macro, sector, recovery


# ── provider 1: Groq (free) ───────────────────────────────────────────────────

def _try_groq(prompt: str) -> dict | None:
    """
    Groq free tier — llama-3.1-8b-instant.
    Free plan: 14,400 requests/day, no credit card required.
    Sign up at https://console.groq.com
    Add the key as secret: GROQ_API_KEY
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        response = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.4,
            },
            timeout=20,
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            macro, sector, recovery = _parse_paragraphs(content)
            return {
                "summary": content,
                "macro_insight": macro,
                "sector_insight": sector,
                "recovery_insight": recovery,
                "source": "🦙 Groq (LLaMA 3.1 · Free)",
            }
    except Exception:
        pass
    return None


# ── provider 2: Google Gemini (free) ─────────────────────────────────────────

def _try_gemini(prompt: str) -> dict | None:
    """
    Google Gemini free tier — gemini-1.5-flash.
    Free plan: 15 RPM / 1 million tokens per day — no billing required.
    Get a free key at https://aistudio.google.com/app/apikey
    Add the key as secret: GEMINI_API_KEY
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={api_key}"
        )
        response = _req.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 600,
                    "temperature": 0.4,
                },
            },
            timeout=25,
        )
        if response.status_code == 200:
            data = response.json()
            content = (
                data["candidates"][0]["content"]["parts"][0]["text"].strip()
            )
            macro, sector, recovery = _parse_paragraphs(content)
            return {
                "summary": content,
                "macro_insight": macro,
                "sector_insight": sector,
                "recovery_insight": recovery,
                "source": "✨ Gemini 1.5 Flash (Free)",
            }
    except Exception:
        pass
    return None


# ── provider 3: OpenAI (paid, optional) ──────────────────────────────────────

def _try_openai(prompt: str) -> dict | None:
    """
    OpenAI GPT-4o-mini — paid plan required.
    Add the key as secret: OPENAI_API_KEY
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
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
            macro, sector, recovery = _parse_paragraphs(content)
            return {
                "summary": content,
                "macro_insight": macro,
                "sector_insight": sector,
                "recovery_insight": recovery,
                "source": "🤖 GPT-4o-mini",
            }
    except Exception:
        pass
    return None


# ── main entry point ──────────────────────────────────────────────────────────

def generate_insights(
    scenario_name: str,
    indices: dict,
    sector_impact: pd.DataFrame,
) -> dict:
    """
    Try each LLM provider in priority order:
      Groq (free) → Gemini (free) → OpenAI (paid) → rule-based (always works)

    Returns dict with keys:
      summary, macro_insight, sector_insight, recovery_insight, source
    """
    prompt = _build_prompt(scenario_name, indices, sector_impact)

    for provider_fn in (_try_groq, _try_gemini, _try_openai):
        result = provider_fn(prompt)
        if result:
            return result

    result = InsightGenerator.generate_scenario_insights(scenario_name, indices, sector_impact)
    result["source"] = "📐 Rule-based (no AI key set)"
    return result
