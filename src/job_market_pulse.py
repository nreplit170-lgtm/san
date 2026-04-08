"""
Job Market Pulse (Feature 2).

Lightweight text analytics on bundled job-posting rows: skill/role frequency,
optional weekly demand trends, and simple salary bands by role.

Replace or extend the CSV with your own exports (Kaggle, ATS, scrapes).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.job_risk_model import SKILL_DEMAND_WEIGHTS

# Extra phrases common in listings (longer first for substring scan order).
EXTRA_SKILL_PHRASES: Tuple[str, ...] = (
    "machine learning engineer",
    "full stack",
    "backend engineer",
    "frontend developer",
    "site reliability",
    "business analyst",
    "java developer",
    "java",
    "spring boot",
    "golang",
    "go engineer",
    "rust",
    "c++",
    "android",
    "ios",
    "swift",
    "kotlin",
    "salesforce",
    "sap",
    "tableau",
    "power bi",
    "looker",
    "etl",
    "snowflake",
    "databricks",
    "terraform",
    "ansible",
    "jenkins",
    "git",
    "agile",
    "scrum",
)

# (substring in title lower, canonical role label) — first match wins.
ROLE_TITLE_RULES: Tuple[Tuple[str, str], ...] = (
    ("data scientist", "Data Scientist"),
    ("machine learning", "ML / AI Engineer"),
    ("ml engineer", "ML / AI Engineer"),
    ("ai engineer", "ML / AI Engineer"),
    ("data engineer", "Data Engineer"),
    ("devops", "DevOps / SRE"),
    ("sre", "DevOps / SRE"),
    ("cloud engineer", "Cloud Engineer"),
    ("security engineer", "Security Engineer"),
    ("cyber security", "Security Engineer"),
    ("product manager", "Product Manager"),
    ("project manager", "Project Manager"),
    ("business analyst", "Business Analyst"),
    ("software engineer", "Software Engineer"),
    ("backend", "Backend Engineer"),
    ("frontend", "Frontend Engineer"),
    ("full stack", "Full-Stack Engineer"),
    ("qa engineer", "QA / Test Engineer"),
    ("test engineer", "QA / Test Engineer"),
    ("sales engineer", "Sales Engineer"),
    ("hr ", "HR / People Ops"),
    ("human resources", "HR / People Ops"),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_jobs_csv_path() -> str:
    return str(_project_root() / "data" / "market_pulse" / "job_postings_sample.csv")


def skill_phrase_list() -> List[str]:
    base = [p for p, _ in sorted(SKILL_DEMAND_WEIGHTS, key=lambda x: -len(x[0]))]
    extra = list(EXTRA_SKILL_PHRASES)
    seen = set()
    out: List[str] = []
    for ph in base + extra:
        if ph not in seen:
            seen.add(ph)
            out.append(ph)
    out.sort(key=len, reverse=True)
    return out


def phrase_in_blob(phrase: str, blob: str) -> bool:
    """Avoid substring false positives (e.g. java ⊂ javascript) for short tokens."""
    if not phrase or not blob:
        return False
    if " " in phrase or len(phrase) >= 5:
        return phrase in blob
    return bool(
        re.search(
            r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])",
            blob,
        )
    )


def classify_role_title(title: str) -> str:
    t = (title or "").lower().strip()
    if not t:
        return "Other / General"
    for needle, label in ROLE_TITLE_RULES:
        if needle in t:
            return label
    return "Other / General"


def prepare_jobs_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "post_date" in out.columns:
        out["post_date"] = pd.to_datetime(out["post_date"], errors="coerce")
    out["role_bucket"] = out.get("job_title", "").map(classify_role_title)
    out["_text"] = (
        out.get("job_title", "").fillna("").astype(str)
        + " "
        + out.get("description", "").fillna("").astype(str)
    ).str.lower()
    return out


def load_job_postings(csv_path: Optional[str] = None) -> pd.DataFrame:
    path = csv_path or default_jobs_csv_path()
    if not os.path.isfile(path):
        return pd.DataFrame()
    raw = pd.read_csv(path)
    return prepare_jobs_df(raw)


def jobs_from_upload(file_obj) -> pd.DataFrame:
    """Streamlit UploadedFile or any file-like with read()."""
    raw = pd.read_csv(file_obj)
    return prepare_jobs_df(raw)


def skill_demand_counts(df: pd.DataFrame, phrases: Optional[Sequence[str]] = None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    phrases = list(phrases or skill_phrase_list())
    counts: Dict[str, int] = {p: 0 for p in phrases}
    for blob in df["_text"]:
        for ph in phrases:
            if phrase_in_blob(ph, blob):
                counts[ph] += 1
    s = pd.Series(counts).sort_values(ascending=False)
    return s[s > 0]


def role_demand_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "role_bucket" not in df.columns:
        return pd.Series(dtype=int)
    return df["role_bucket"].value_counts()


def weekly_skill_trends(
    df: pd.DataFrame,
    top_n_skills: int = 5,
    phrases: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Rows = week period (start date as string), columns = skill, values = mention count.
    """
    if df.empty or "post_date" not in df.columns or df["post_date"].isna().all():
        return pd.DataFrame()

    phrases = list(phrases or skill_phrase_list())
    overall = skill_demand_counts(df, phrases)
    top_skills = list(overall.head(top_n_skills).index)
    if not top_skills:
        return pd.DataFrame()

    df2 = df.dropna(subset=["post_date"]).copy()
    df2["week"] = df2["post_date"].dt.to_period("W").apply(lambda p: p.start_time.date())

    records = []
    for _, row in df2.iterrows():
        blob = row["_text"]
        w = row["week"]
        for sk in top_skills:
            if phrase_in_blob(sk, blob):
                records.append({"week": w, "skill": sk})

    if not records:
        return pd.DataFrame()
    tall = pd.DataFrame(records)
    pivot = tall.groupby(["week", "skill"]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()
    pivot.index.name = "week"
    return pivot


def skill_momentum(
    df: pd.DataFrame,
    top_n_skills: int = 15,
    phrases: Optional[Sequence[str]] = None,
    recent_weeks: int = 4,
) -> pd.DataFrame:
    """
    For each top skill, compute total mentions in the most-recent N weeks vs
    all earlier weeks to produce a momentum label and numeric delta.

    Returns DataFrame with columns: skill, recent, earlier, delta_pct, momentum
    """
    if df.empty or "post_date" not in df.columns or df["post_date"].isna().all():
        return pd.DataFrame()

    phrases = list(phrases or skill_phrase_list())
    overall = skill_demand_counts(df, phrases)
    top_skills = list(overall.head(top_n_skills).index)
    if not top_skills:
        return pd.DataFrame()

    df2 = df.dropna(subset=["post_date"]).copy()
    df2["week"] = df2["post_date"].dt.to_period("W").apply(lambda p: p.start_time.date())
    all_weeks = sorted(df2["week"].unique())
    if len(all_weeks) < 2:
        return pd.DataFrame()

    cutoff = all_weeks[-min(recent_weeks, len(all_weeks) - 1)]
    recent_df = df2[df2["week"] >= cutoff]
    earlier_df = df2[df2["week"] < cutoff]

    rows = []
    for sk in top_skills:
        r = sum(1 for blob in recent_df["_text"] if phrase_in_blob(sk, blob))
        e = sum(1 for blob in earlier_df["_text"] if phrase_in_blob(sk, blob))
        if e == 0 and r == 0:
            continue
        e_norm = max(e, 1)
        delta_pct = round((r - e_norm) / e_norm * 100, 1)
        if delta_pct >= 15:
            momentum = "Rising"
        elif delta_pct <= -15:
            momentum = "Declining"
        else:
            momentum = "Stable"
        rows.append({"skill": sk, "recent": r, "earlier": e, "delta_pct": delta_pct, "momentum": momentum})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("delta_pct", ascending=False).reset_index(drop=True)


def location_demand_counts(df: pd.DataFrame, top_n: int = 12) -> pd.Series:
    """Count postings by location."""
    if df.empty or "location" not in df.columns:
        return pd.Series(dtype=int)
    counts = df["location"].dropna().astype(str).value_counts().head(top_n)
    return counts[counts > 0]


def skill_gap_analysis(
    df: pd.DataFrame,
    user_skills_text: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compare top-demanded skills in postings against user's skill set.
    Returns DataFrame with skill, demand_count, have (bool).
    """
    top_skills = skill_demand_counts(df).head(top_n)
    if top_skills.empty:
        return pd.DataFrame()
    user_blob = user_skills_text.lower()
    rows = []
    for sk, cnt in top_skills.items():
        have = phrase_in_blob(sk, user_blob)
        rows.append({"Skill": sk, "Demand (postings)": int(cnt), "You have it": have})
    return pd.DataFrame(rows)


def salary_summary_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """Uses salary_min_lpa / salary_max_lpa midpoint when both present."""
    if df.empty or "role_bucket" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    if "salary_min_lpa" not in d.columns or "salary_max_lpa" not in d.columns:
        return pd.DataFrame()
    smin = pd.to_numeric(d["salary_min_lpa"], errors="coerce")
    smax = pd.to_numeric(d["salary_max_lpa"], errors="coerce")
    mid = (smin + smax) / 2.0
    d["_salary_mid"] = mid
    sub = d.dropna(subset=["_salary_mid"])
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("role_bucket")["_salary_mid"].agg(["median", "mean", "count"])
    g = g.sort_values("median", ascending=False)
    g.columns = ["median_lpa", "mean_lpa", "postings"]
    return g.round(2)
