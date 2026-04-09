"""
live_insights.py
Rule-based insight generator for live World Bank labor data.
Works without any API key — pure interpretation of the numbers.
"""
from __future__ import annotations
import pandas as pd


_INDIA_HISTORICAL_PEAK = 9.5     # recorded peak (2000-01, PLFS)
_YOUTH_MULTIPLIER_WARN  = 4.0    # youth rate > 4× adult → structural concern
_LFP_INDIA_BENCHMARK    = 55.0   # World Bank 2023 India LFP (%)
_VULN_WARN_THRESHOLD    = 70.0   # >70% vulnerable employment is high


def _latest(series_map: dict, label: str):
    """Return (latest_value, latest_year, prev_value | None) from wb_data dict."""
    s = series_map.get(label)
    if s is None or s.empty:
        return None, None, None
    latest_val  = float(s.iloc[-1]["Value"])
    latest_year = int(s.iloc[-1]["Year"])
    prev_val    = float(s.iloc[-2]["Value"]) if len(s) >= 2 else None
    return latest_val, latest_year, prev_val


def generate_labor_market_insights(wb_data: dict) -> list[str]:
    """
    Takes the dict returned by fetch_labor_market_pulse() and returns
    a list of insight strings in plain English.

    Each string is a single bullet-point insight ready to render directly.
    """
    insights = []

    ue_val, ue_yr, ue_prev    = _latest(wb_data, "Unemployment Rate (%)")
    youth_val, _, _           = _latest(wb_data, "Youth Unemployment 15-24 (%)")
    female_val, _, _          = _latest(wb_data, "Female Unemployment (%)")
    male_val, _, _            = _latest(wb_data, "Male Unemployment (%)")
    lfp_val, _, lfp_prev      = _latest(wb_data, "Labor Force Participation (%)")
    vuln_val, _, _            = _latest(wb_data, "Vulnerable Employment (%)")
    lt_val, _, _              = _latest(wb_data, "Long-Term Unemployment (%)")

    # ── Headline unemployment
    if ue_val is not None:
        if ue_val <= 4.0:
            insights.append(
                f"India's unemployment rate stands at **{ue_val:.1f}%** ({ue_yr}) — "
                f"historically low, but the official figure understates underemployment due to "
                f"the large informal sector."
            )
        elif ue_val <= 7.0:
            insights.append(
                f"India's unemployment rate is **{ue_val:.1f}%** ({ue_yr}), in the moderate range. "
                f"This represents structural friction unemployment in a rapidly urbanising economy."
            )
        else:
            insights.append(
                f"India's unemployment rate is **{ue_val:.1f}%** ({ue_yr}) — elevated. "
                f"The historical peak was {_INDIA_HISTORICAL_PEAK}% (2000–01). "
                f"Current levels indicate structural stress in the labour market."
            )

        if ue_prev is not None:
            chg = round(ue_val - ue_prev, 2)
            if abs(chg) >= 0.3:
                direction = "increased" if chg > 0 else "improved (fallen)"
                insights.append(
                    f"The rate has **{direction} by {abs(chg)} pp** year-on-year — "
                    f"{'a deterioration worth monitoring' if chg > 0 else 'a positive signal for the labour market'}."
                )

    # ── Youth vs adult gap
    if youth_val is not None and ue_val is not None and ue_val > 0:
        ratio = round(youth_val / ue_val, 1)
        if ratio >= _YOUTH_MULTIPLIER_WARN:
            insights.append(
                f"Youth unemployment ({youth_val:.1f}%) is **{ratio}× the overall rate** — "
                f"a severe structural mismatch between education output and labour demand. "
                f"Entry-level job creation and skilling investment are priority policy levers."
            )
        else:
            insights.append(
                f"Youth unemployment ({youth_val:.1f}%) is {ratio}× the overall rate — "
                f"within the typical range for a developing economy, though youth skills "
                f"alignment with industry demand remains a key challenge."
            )

    # ── Gender gap
    if female_val is not None and male_val is not None:
        gap = round(female_val - male_val, 1)
        if gap > 2.0:
            insights.append(
                f"Female unemployment ({female_val:.1f}%) is **{gap} pp higher** than male ({male_val:.1f}%) — "
                f"indicating gender-based labour market barriers. Reducing this gap could "
                f"significantly expand India's productive workforce."
            )
        elif gap < -1.0:
            insights.append(
                f"Male unemployment ({male_val:.1f}%) is marginally above female ({female_val:.1f}%) — "
                f"an unusual pattern; may reflect low female labour force participation "
                f"masking the true female unemployment situation."
            )
        else:
            insights.append(
                f"Gender unemployment gap is narrow: female {female_val:.1f}% vs male {male_val:.1f}%. "
                f"However, low female LFP can make official unemployment rates misleadingly similar."
            )

    # ── Labour force participation
    if lfp_val is not None:
        if lfp_val < 50.0:
            insights.append(
                f"Labour force participation at **{lfp_val:.1f}%** is well below the global "
                f"average (~60%). Low participation — driven especially by female non-participation "
                f"— represents a large untapped potential in India's workforce."
            )
        elif lfp_val < _LFP_INDIA_BENCHMARK:
            insights.append(
                f"LFP at **{lfp_val:.1f}%** remains below India's benchmark of "
                f"{_LFP_INDIA_BENCHMARK}%. Improving this — particularly through female "
                f"workforce inclusion — would accelerate economic growth."
            )
        else:
            insights.append(
                f"Labour force participation ({lfp_val:.1f}%) is at or above India's benchmark, "
                f"suggesting good workforce engagement. Sustaining this while improving "
                f"job quality is the next policy challenge."
            )

    # ── Vulnerable employment
    if vuln_val is not None:
        if vuln_val >= _VULN_WARN_THRESHOLD:
            insights.append(
                f"**{vuln_val:.1f}% of employment is vulnerable** (informal, own-account, or unpaid) — "
                f"the majority of India's workers lack social protections, stable income, "
                f"or formal contracts. Formalisation policies are critical."
            )
        else:
            insights.append(
                f"Vulnerable employment at {vuln_val:.1f}% shows a slight improvement over India's "
                f"historically high informality rates. Formalisation progress is encouraging "
                f"but still has a long way to go."
            )

    # ── Long-term unemployment
    if lt_val is not None and lt_val > 20.0:
        insights.append(
            f"Long-term unemployment at **{lt_val:.1f}%** of total unemployed suggests "
            f"skill obsolescence and low job mobility. Active labour market policies "
            f"(retraining, job matching) would directly address this group."
        )

    return insights


def generate_sector_insights(sector_df: pd.DataFrame) -> list[str]:
    """
    Takes the DataFrame returned by fetch_sector_indicators() and returns
    insights about sector structure.
    """
    insights = []
    if sector_df.empty:
        return ["Sector data unavailable from World Bank API at this time."]

    df = sector_df.dropna(subset=["Employment_Share"])
    if df.empty:
        return ["Employment share data unavailable — GDP share data still shown above."]

    top_emp = df.sort_values("Employment_Share", ascending=False).iloc[0]
    insights.append(
        f"**{top_emp['Sector']}** employs the largest share of India's workforce "
        f"(**{top_emp['Employment_Share']:.1f}%**) — this sector's health directly determines "
        f"livelihood for the most Indians."
    )

    # Agriculture-services gap
    agri = df[df["Sector"] == "Agriculture"]
    svcs = df[df["Sector"] == "Services"]
    if not agri.empty and not svcs.empty:
        agri_emp = agri.iloc[0]["Employment_Share"]
        svcs_emp = svcs.iloc[0]["Employment_Share"]
        insights.append(
            f"Agriculture ({agri_emp:.1f}% of employment) remains far larger than its GDP contribution — "
            f"while Services ({svcs_emp:.1f}% of employment) generates a disproportionately "
            f"higher share of GDP, highlighting a structural productivity gap."
        )

    gdp_df = sector_df.dropna(subset=["GDP_Share"])
    if not gdp_df.empty:
        top_gdp = gdp_df.sort_values("GDP_Share", ascending=False).iloc[0]
        insights.append(
            f"By GDP contribution, **{top_gdp['Sector']}** leads at {top_gdp['GDP_Share']:.1f}% — "
            f"diversifying GDP while supporting employment in lower-productivity sectors "
            f"is India's central structural challenge."
        )

    return insights


def generate_forecast_insights(hist_df: pd.DataFrame, fc_df: pd.DataFrame) -> list[str]:
    """
    Generate insights about a real-data evidence-based forecast.
    hist_df: historical WB data (Year, Unemployment_Rate)
    fc_df: forecast output (Year, Predicted_Unemployment, Lower_80, Upper_80)
    """
    insights = []
    if hist_df.empty or fc_df.empty:
        return []

    current = round(float(hist_df["Unemployment_Rate"].iloc[-1]), 2)
    current_yr = int(hist_df["Year"].iloc[-1])
    fc_end = round(float(fc_df["Predicted_Unemployment"].iloc[-1]), 2)
    fc_end_yr = int(fc_df["Year"].iloc[-1])
    lower = round(float(fc_df["Lower_80"].iloc[-1]), 2)
    upper = round(float(fc_df["Upper_80"].iloc[-1]), 2)

    chg = round(fc_end - current, 2)
    direction = "rise to" if chg > 0.3 else ("fall to" if chg < -0.3 else "remain near")

    insights.append(
        f"Based on **{len(hist_df)} years of real World Bank data**, the ensemble model forecasts "
        f"India's unemployment to **{direction} {fc_end}%** by {fc_end_yr} "
        f"(80% confidence band: {lower}%–{upper}%)."
    )

    # Trend context
    recent_10 = hist_df.tail(10)
    trend_slope = (recent_10["Unemployment_Rate"].iloc[-1] - recent_10["Unemployment_Rate"].iloc[0]) / 9
    if trend_slope < -0.1:
        insights.append(
            f"The 10-year trend has been **declining** (~{abs(round(trend_slope, 2))} pp/year), "
            f"which anchors the forecast. Mean-reversion limits how far this can continue without "
            f"structural improvements in job creation."
        )
    elif trend_slope > 0.1:
        insights.append(
            f"The 10-year trend has been **rising** (~{round(trend_slope, 2)} pp/year). "
            f"The model applies mean-reversion to moderate this — but sustained increases "
            f"would indicate structural weakening of India's labour market."
        )
    else:
        insights.append(
            f"The 10-year trend has been **broadly stable**, with the model projecting "
            f"mean-reverting fluctuations rather than a directional move. "
            f"Short-term shocks (policy, global) remain the key variable to watch."
        )

    band_width = round(upper - lower, 2)
    insights.append(
        f"The {band_width} pp uncertainty band reflects historical volatility — "
        f"India's unemployment data shows {round(hist_df['Unemployment_Rate'].std(), 2)} pp "
        f"standard deviation over the full series. Global shocks (like COVID-19 in 2020) "
        f"can push outcomes well outside normal bounds."
    )

    return insights
