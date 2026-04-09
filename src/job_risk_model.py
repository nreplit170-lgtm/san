"""
Job unemployment risk predictor (Feature 1).

Trains a logistic regression on synthetic-but-structured data so the model
learns sensible relationships: stronger skills, education, experience,
industry growth, and better locations → lower estimated risk.

The UI calls this module directly; it does not depend on the FastAPI stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "skill_demand_score",
    "industry_growth",
    "experience_years",
    "education_level",
    "location_risk_tier",
]

# Keyword → demand weight (0–1). Longer phrases matched first if we sort by length.
SKILL_DEMAND_WEIGHTS: List[Tuple[str, float]] = sorted(
    [
        ("machine learning", 0.98),
        ("deep learning", 0.97),
        ("data science", 0.96),
        ("cloud computing", 0.94),
        ("aws", 0.92),
        ("azure", 0.91),
        ("kubernetes", 0.93),
        ("devops", 0.90),
        ("python", 0.88),
        ("sql", 0.85),
        ("javascript", 0.82),
        ("react", 0.84),
        ("node", 0.83),
        ("cybersecurity", 0.93),
        ("product management", 0.80),
        ("project management", 0.72),
        ("excel", 0.55),
        ("communication", 0.65),
        ("jquery", 0.35),
        ("php", 0.58),
        ("manual testing", 0.50),
        ("data entry", 0.42),
    ],
    key=lambda x: -len(x[0]),
)

EDUCATION_LEVELS = [
    "Less than high school",
    "High school / diploma",
    "Bachelor's degree",
    "Master's degree",
    "Doctorate / professional",
]

LOCATION_OPTIONS = [
    "Metro / Tier-1 city",
    "Tier-2 city",
    "Smaller town / rural",
]

INDUSTRY_GROWTH = {
    "Technology / software": 0.92,
    "Healthcare / biotech": 0.88,
    "Financial services / fintech": 0.82,
    "Renewable energy / climate": 0.86,
    "Education / edtech": 0.72,
    "Retail / e-commerce ops": 0.62,
    "Manufacturing (traditional)": 0.55,
    "Hospitality / tourism": 0.48,
    "Other / not listed": 0.60,
}


def _location_risk_tier(label: str) -> float:
    if label == LOCATION_OPTIONS[0]:
        return 0.0
    if label == LOCATION_OPTIONS[1]:
        return 1.0
    return 2.0


def parse_skills(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[,;\n]+", str(text).lower())
    return [p.strip() for p in parts if p.strip()]


def compute_skill_demand_score(skills: List[str]) -> Tuple[float, List[str]]:
    """
    Returns score in [0, 1] and list of matched **high-demand** keywords (for UI).
    Phrases below STRONG_SKILL_THRESHOLD still affect the mean score but are not listed as "in-demand".
    """
    STRONG = 0.68
    if not skills:
        return 0.45, []

    blob = " ".join(skills)
    weights: List[float] = []
    strong_matched: List[str] = []
    for phrase, w in SKILL_DEMAND_WEIGHTS:
        if phrase in blob:
            weights.append(w)
            if w >= STRONG:
                strong_matched.append(phrase)
    if not weights:
        generic = min(1.0, 0.35 + 0.02 * len(skills))
        return generic, []

    return float(np.clip(np.mean(weights), 0.0, 1.0)), strong_matched


def build_feature_row(
    skills_text: str,
    education_label: str,
    experience_years: int,
    location_label: str,
    industry_label: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    skills = parse_skills(skills_text)
    skill_score, matched = compute_skill_demand_score(skills)
    try:
        edu = float(EDUCATION_LEVELS.index(education_label))
    except ValueError:
        edu = 2.0
    ind = float(INDUSTRY_GROWTH.get(industry_label, 0.6))
    loc = _location_risk_tier(location_label)
    exp = float(np.clip(experience_years, 0, 40))

    row = np.array(
        [[skill_score, ind, exp, edu, loc]],
        dtype=np.float64,
    )
    meta = {
        "parsed_skills": skills,
        "matched_high_demand": matched,
        "skill_demand_score": skill_score,
        "industry_growth": ind,
        "experience_years": exp,
        "education_level": edu,
        "location_risk_tier": loc,
    }
    return row, meta


def _synthetic_dataset(n_samples: int = 3500, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    skill = rng.uniform(0.15, 1.0, n_samples)
    ind = rng.uniform(0.35, 0.98, n_samples)
    exp = rng.integers(0, 36, n_samples).astype(np.float64)
    edu = rng.integers(0, 5, n_samples).astype(np.float64)
    loc = rng.integers(0, 3, n_samples).astype(np.float64)

    edu_norm = edu / 4.0
    exp_norm = np.clip(exp / 25.0, 0.0, 1.0)
    loc_norm = loc / 2.0

    logit = (
        2.1
        - 2.8 * skill
        - 1.9 * ind
        - 0.55 * edu_norm
        - 0.9 * exp_norm
        + 1.1 * loc_norm
        + rng.normal(0.0, 0.55, n_samples)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_samples) < p).astype(np.int32)

    X = np.column_stack([skill, ind, exp, edu, loc])
    return X, y


@dataclass
class JobRiskResult:
    high_risk_probability_pct: float
    risk_level: str
    features: Dict[str, Any]
    reasons: List[str]
    suggestions: List[str]
    # Optional — callers must None-check before calling .items().
    contributions: Optional[Dict[str, float]] = None


def _risk_level_from_prob(p: float) -> str:
    if p >= 0.62:
        return "High"
    if p >= 0.35:
        return "Medium"
    return "Low"


def _train_pipeline() -> Pipeline:
    X, y = _synthetic_dataset()
    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


_PIPE: Optional[Pipeline] = None

# Precompute feature means once at module load — used for contribution attribution.
# _synthetic_dataset() is deterministic (seed=42), so means are always identical;
# recomputing them inside predict_job_risk() every call was pure waste.
_FEATURE_MEANS: Optional[np.ndarray] = None


def get_pipeline() -> Pipeline:
    global _PIPE, _FEATURE_MEANS
    if _PIPE is None:
        _PIPE = _train_pipeline()
        X_all, _ = _synthetic_dataset()
        _FEATURE_MEANS = X_all.mean(axis=0)
    return _PIPE


def _get_feature_means() -> np.ndarray:
    get_pipeline()   # ensures _FEATURE_MEANS is populated
    return _FEATURE_MEANS


def _linear_contributions(
    pipe: Pipeline, X_row: np.ndarray, feature_means: np.ndarray
) -> Dict[str, float]:
    clf: LogisticRegression = pipe.named_steps["clf"]
    scale: StandardScaler = pipe.named_steps["scale"]
    Xs = scale.transform(X_row)
    mu_s = scale.transform(feature_means.reshape(1, -1))
    diff = (Xs - mu_s).ravel()
    coef = clf.coef_.ravel()
    return {FEATURE_NAMES[i]: float(diff[i] * coef[i]) for i in range(len(FEATURE_NAMES))}


def predict_job_risk(
    skills_text: str,
    education_label: str,
    experience_years: int,
    location_label: str,
    industry_label: str,
) -> JobRiskResult:
    pipe = get_pipeline()
    X_row, meta = build_feature_row(
        skills_text, education_label, experience_years, location_label, industry_label
    )

    proba = float(pipe.predict_proba(X_row)[0, 1])
    level = _risk_level_from_prob(proba)

    # Use precomputed means — no need to regenerate 3,500-sample dataset each call.
    means = _get_feature_means()
    contribs = _linear_contributions(pipe, X_row, means)
    sorted_c = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    reasons: List[str] = []
    for name, c in sorted_c[:4]:
        if name == "skill_demand_score":
            if c > 0.15:
                reasons.append("Skill profile aligns weakly with high-demand areas (raises modeled risk).")
            elif c < -0.15:
                reasons.append("Strong match to in-demand skills (lowers modeled risk).")
        elif name == "industry_growth":
            if c > 0.12:
                reasons.append("Selected industry has below-average growth in the model (raises risk).")
            elif c < -0.12:
                reasons.append("Industry growth outlook supports employability (lowers risk).")
        elif name == "experience_years":
            if c > 0.1:
                reasons.append("Limited years of experience contribute to higher modeled risk.")
            elif c < -0.1:
                reasons.append("Experience depth reduces modeled unemployment risk.")
        elif name == "education_level":
            if c > 0.1:
                reasons.append("Formal education level is a drag on the risk score in this profile.")
            elif c < -0.1:
                reasons.append("Higher education level reduces modeled risk.")
        elif name == "location_risk_tier":
            if c > 0.12:
                reasons.append("Location tier suggests fewer local opportunities in the model.")
            elif c < -0.12:
                reasons.append("Location tier is favorable for job market access.")

    if not reasons:
        reasons.append("Risk is near the model’s average for similar synthetic profiles.")

    suggestions: List[str] = []
    if proba >= 0.35:
        if float(meta["skill_demand_score"]) < 0.72:
            suggestions.append("Add skills that appear frequently in growing roles (e.g. cloud, data, security).")
        if meta["education_level"] < 3:
            suggestions.append("Consider certifications or degree progress in a high-growth domain.")
        if meta["location_risk_tier"] >= 1.5:
            suggestions.append("Explore remote-first roles or hubs with stronger hiring in your field.")
        if meta["industry_growth"] < 0.65:
            suggestions.append("Research adjacent industries with higher hiring momentum.")
    else:
        suggestions.append("Maintain skills and monitor industry shifts; your profile looks comparatively resilient.")

    return JobRiskResult(
        high_risk_probability_pct=round(proba * 100.0, 1),
        risk_level=level,
        features=meta,
        reasons=reasons[:5],
        suggestions=suggestions[:5],
        contributions={k: round(v, 4) for k, v in contribs.items()},
    )


def industry_risk_comparison(
    skills_text: str,
    education_label: str,
    experience_years: int,
    location_label: str,
) -> List[Dict[str, Any]]:
    """Run the model across all industries to show where the user fits best."""
    rows = []
    for ind_label in INDUSTRY_GROWTH:
        r = predict_job_risk(skills_text, education_label, experience_years, location_label, ind_label)
        rows.append({
            "Industry": ind_label,
            "Risk (%)": r.high_risk_probability_pct,
            "Level": r.risk_level,
        })
    return sorted(rows, key=lambda x: x["Risk (%)"])


def what_if_improve_skills(
    skills_text: str,
    education_label: str,
    experience_years: int,
    location_label: str,
    industry_label: str,
    extra_skills_text: str,
) -> Tuple[JobRiskResult, JobRiskResult, float]:
    base = predict_job_risk(
        skills_text, education_label, experience_years, location_label, industry_label
    )
    merged = ", ".join(s for s in [skills_text.strip(), extra_skills_text.strip()] if s)
    improved = predict_job_risk(
        merged, education_label, experience_years, location_label, industry_label
    )
    delta = round(
        improved.high_risk_probability_pct - base.high_risk_probability_pct, 1
    )
    return base, improved, delta
