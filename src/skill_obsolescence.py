"""
skill_obsolescence.py
Detects declining and emerging skills from job-posting time-series data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def detect_skill_obsolescence(
    df: pd.DataFrame,
    date_col: str = "date",
    skills_col: str = "skills",
    min_mentions: int = 3,
    trend_threshold: float = 0.05,
) -> Dict:
    """
    Analyses skill mention frequency over time to classify skills as:
      - emerging:  positive trend in demand
      - declining: negative trend in demand
      - stable:    flat demand

    Parameters
    ----------
    df              : DataFrame with at least a date column and a skills text column
    date_col        : column containing posting date (any parseable date format)
    skills_col      : column containing comma-separated or space-separated skill text
    min_mentions    : minimum total mentions required to classify a skill
    trend_threshold : minimum |slope| (per period) to classify as emerging/declining

    Returns
    -------
    dict with keys: emerging, declining, stable, skill_trends (DataFrame)
    """
    if df.empty or skills_col not in df.columns:
        return {"emerging": [], "declining": [], "stable": [], "skill_trends": pd.DataFrame()}

    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df["_period"] = df[date_col].dt.to_period("Q")
    else:
        df["_period"] = "Q1"

    COMMON_SKILLS = [
        "python", "machine learning", "sql", "java", "javascript", "react",
        "aws", "docker", "kubernetes", "data analysis", "excel", "power bi",
        "tableau", "tensorflow", "pytorch", "nlp", "cloud", "devops",
        "git", "agile", "communication", "leadership", "project management",
        "cybersecurity", "blockchain", "iot", "flutter", "golang",
        "jquery", "php", "hadoop", "spark", "r language",
    ]

    rows = []
    for _, row in df.iterrows():
        blob = str(row.get(skills_col, "")).lower()
        period = row["_period"]
        for skill in COMMON_SKILLS:
            if skill in blob:
                rows.append({"skill": skill, "period": str(period)})

    if not rows:
        return {"emerging": [], "declining": [], "stable": [], "skill_trends": pd.DataFrame()}

    mentions_df = pd.DataFrame(rows)
    pivot = mentions_df.groupby(["period", "skill"]).size().reset_index(name="count")
    pivot = pivot.pivot(index="period", columns="skill", values="count").fillna(0)

    periods = np.arange(len(pivot))

    emerging, declining, stable = [], [], []
    trend_rows = []

    for skill in pivot.columns:
        series = pivot[skill].values
        total = int(series.sum())
        if total < min_mentions:
            continue

        if len(series) >= 2:
            slope = float(np.polyfit(periods, series, 1)[0])
        else:
            slope = 0.0

        if slope > trend_threshold:
            category = "emerging"
            emerging.append(skill)
        elif slope < -trend_threshold:
            category = "declining"
            declining.append(skill)
        else:
            category = "stable"
            stable.append(skill)

        trend_rows.append({
            "skill": skill,
            "total_mentions": total,
            "slope": round(slope, 3),
            "category": category,
        })

    trend_df = pd.DataFrame(trend_rows).sort_values("slope", ascending=False) if trend_rows else pd.DataFrame()

    return {
        "emerging": emerging,
        "declining": declining,
        "stable": stable,
        "skill_trends": trend_df,
    }
