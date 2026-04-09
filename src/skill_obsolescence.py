"""
skill_obsolescence.py

Detects declining and emerging skills from job-posting time-series data using
linear regression on log1p-transformed mention counts (bucketted by month or week).

Returns:
  summary_df  — one row per skill with slope, p_value, category, first/last mentions,
                estimated_months_to_fade / estimated_months_to_emerge
  pivot_df    — rows=bucket, columns=skills, values=mention counts
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd

from src.job_market_pulse import phrase_in_blob, skill_phrase_list


def _bucket_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Build a skill × bucket mention pivot from a prepared jobs DataFrame."""
    if df.empty or "post_date" not in df.columns or df["post_date"].isna().all():
        return pd.DataFrame()

    phrases = skill_phrase_list()
    df2 = df.dropna(subset=["post_date"]).copy()

    if freq == "W":
        df2["bucket"] = df2["post_date"].dt.to_period("W").apply(
            lambda p: str(p.start_time.date())
        )
    else:
        df2["bucket"] = df2["post_date"].dt.to_period("M").apply(
            lambda p: str(p.start_time.date())[:7]
        )

    records = []
    for _, row in df2.iterrows():
        blob = row.get("_text", "")
        b = row["bucket"]
        for ph in phrases:
            if phrase_in_blob(ph, blob):
                records.append({"bucket": b, "skill": ph})

    if not records:
        return pd.DataFrame()

    tall = pd.DataFrame(records)
    pivot = (
        tall.groupby(["bucket", "skill"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    pivot.index.name = "bucket"
    return pivot


def _months_to_threshold(
    last_val: float,
    slope_raw: float,
    threshold: float,
    freq: str,
    direction: str,
) -> float | None:
    """Estimate periods until mention count crosses threshold; convert to months."""
    if abs(slope_raw) < 1e-9:
        return None
    if direction == "fade" and slope_raw >= 0:
        return None
    if direction == "emerge" and slope_raw <= 0:
        return None
    steps = (threshold - last_val) / slope_raw
    if steps < 0:
        return None
    weeks_per_step = 4.0 if freq == "M" else 1.0
    months = steps * weeks_per_step / 4.0
    return round(months, 1)


def detect_skill_obsolescence(
    df: pd.DataFrame,
    freq: str = "M",
    top_k: int = 12,
    min_total_mentions: int = 6,
    alpha: float = 0.05,
    slope_threshold_log: float = 0.02,
    category_min_change_ratio: float = 1.8,
    fade_threshold_mentions: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    df                        : prepared jobs DataFrame (must have _text, post_date)
    freq                      : 'M' (monthly) or 'W' (weekly)
    top_k                     : analyse the top-K skills by total mention count
    min_total_mentions        : discard skills below this total
    alpha                     : p-value threshold for the linregress significance test
    slope_threshold_log       : minimum |slope| on log1p(counts) to be flagged
    category_min_change_ratio : last/first ratio needed to label Emerging/Declining
    fade_threshold_mentions   : mentions-per-bucket target used for fade estimate

    Returns
    -------
    (summary_df, pivot_df)
    """
    from scipy import stats as scipy_stats

    pivot = _bucket_series(df, freq)
    if pivot.empty:
        return pd.DataFrame(), pd.DataFrame()

    totals = pivot.sum().sort_values(ascending=False)
    top_skills = totals[totals >= min_total_mentions].head(top_k).index.tolist()
    if not top_skills:
        return pd.DataFrame(), pivot

    pivot = pivot[top_skills]
    t = np.arange(len(pivot), dtype=float)

    # Compute a consistent emergence target BEFORE the skill loop.
    # Previously, each skill's target was last_v * 2, meaning a skill at 50
    # mentions had to reach 100 while a skill at 5 only needed to reach 10 —
    # making estimated months structurally incomparable across skills.
    # Using the 75th-percentile last-bucket value gives every skill the same bar:
    # "reach top-quartile demand among all analysed skills."
    last_vals_all = [float(pivot[sk].values[-1]) for sk in top_skills]
    target_emerge = float(np.percentile(last_vals_all, 75)) if last_vals_all else 0.0

    rows = []

    for skill in top_skills:
        raw = pivot[skill].values.astype(float)
        total = int(raw.sum())
        first_v = float(raw[0])
        last_v = float(raw[-1])

        log_vals = np.log1p(raw)
        if len(t) >= 3:
            result = scipy_stats.linregress(t, log_vals)
            slope_log = float(result.slope)
            p_val = float(result.pvalue)
        elif len(t) == 2:
            slope_log = float(log_vals[1] - log_vals[0])
            p_val = 1.0
        else:
            slope_log = 0.0
            p_val = 1.0

        slope_raw = float(np.polyfit(t, raw, 1)[0]) if len(t) >= 2 else 0.0

        significant = p_val < alpha and abs(slope_log) >= slope_threshold_log

        if significant and slope_log > 0:
            ratio_ok = (last_v >= first_v * category_min_change_ratio) if first_v > 0 else (last_v > 0)
            category = "Emerging" if ratio_ok else "Stable"
        elif significant and slope_log < 0:
            ratio_ok = (last_v <= first_v / category_min_change_ratio) if last_v > 0 else True
            category = "Declining" if ratio_ok else "Stable"
        else:
            category = "Stable"

        months_fade = _months_to_threshold(last_v, slope_raw, fade_threshold_mentions, freq, "fade")
        # Use the pre-computed 75th-percentile target for a consistent emergence bar.
        months_emerge = _months_to_threshold(last_v, slope_raw, target_emerge, freq, "emerge")

        rows.append({
            "skill": skill,
            "category": category,
            "total_mentions": total,
            "first_mentions": int(first_v),
            "last_mentions": int(last_v),
            "slope_mentions_per_step": round(slope_raw, 3),
            "slope_log": round(slope_log, 4),
            "p_value": round(p_val, 4),
            "estimated_months_to_fade": months_fade,
            "estimated_months_to_emerge": months_emerge,
        })

    summary_df = (
        pd.DataFrame(rows)
        .sort_values(["category", "p_value"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return summary_df, pivot
