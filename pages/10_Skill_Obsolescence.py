"""
Page 10 — Skill Obsolescence Detector (Feature 4)

Computes time-series skill mention trends from a job-postings CSV and flags
skills that are statistically trending up (emerging) or down (declining).
"""

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.job_market_pulse import jobs_from_upload, load_job_postings
from src.skill_obsolescence import detect_skill_obsolescence
from src.ui_helpers import DARK_CSS, plotly_dark_layout


st.set_page_config(page_title="Skill Obsolescence | UIP", page_icon="📉", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📉 Skill Obsolescence")
    st.caption("Trend detection from posting text (post_date) — not hardcoded.")
    st.markdown("---")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/2_Simulator.py", label="🧪 Simulator")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")


st.markdown(
    """
<div class="page-hero">
  <div class="hero-title">📉 Skill Obsolescence Detector</div>
  <div class="hero-subtitle">
    Detect declining vs emerging skills using mention trends over time from your posting CSV.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

upload = st.file_uploader(
    "Optional: upload historical job postings CSV",
    type=["csv"],
    help="Expected columns: post_date, job_title, description. "
    "Other columns are ignored.",
)

if upload is None:
    df = load_job_postings()
    src_label = "Default sample CSV (data/market_pulse/job_postings_sample.csv)"
else:
    df = jobs_from_upload(io.BytesIO(upload.getvalue()))
    src_label = f"Uploaded CSV: {upload.name}"

if df.empty:
    st.error("No job postings loaded. Ensure your CSV has a parsable `post_date` column.")
    st.stop()

with st.expander("Detection settings", expanded=False):
    freq = st.selectbox("Trend bucket size", ["M", "W"], index=0, format_func=lambda x: "Monthly" if x == "M" else "Weekly")
    top_k = st.slider("Analyze top skills (by total mentions)", 6, 30, 12, 1)
    min_total_mentions = st.slider("Min mentions to consider a skill", 1, 50, 6, 1)
    alpha = st.selectbox("Significance level (alpha)", [0.10, 0.05, 0.01], index=1)
    slope_threshold_log = st.slider(
        "Min trend strength (|slope| on log1p counts)",
        0.005,
        0.20,
        0.02,
        0.005,
        help="Higher = fewer skills flagged; tuned for noisy job text.",
    )
    category_min_change_ratio = st.slider(
        "Min relative change to label 'Emerging'/'Declining'",
        1.2,
        3.5,
        1.8,
        0.1,
        help="For emerging: last mentions >= first * ratio. "
        "For declining: last mentions <= first / ratio.",
    )
    fade_threshold_mentions = st.slider(
        "Fade threshold (mentions per bucket)",
        0,
        3,
        1,
        1,
        help="Used to estimate how long until mentions drop to this level (heuristic).",
    )

run = st.button("🔍 Detect emerging & declining skills", use_container_width=True)

if not run:
    st.caption(src_label)
    st.info("Click the detect button to run trend analysis.")
    st.stop()

summary_df, pivot = detect_skill_obsolescence(
    df=df,
    freq=freq,
    top_k=top_k,
    min_total_mentions=min_total_mentions,
    alpha=float(alpha),
    slope_threshold_log=float(slope_threshold_log),
    category_min_change_ratio=float(category_min_change_ratio),
    fade_threshold_mentions=int(fade_threshold_mentions),
)

st.caption(src_label)

if summary_df.empty or pivot.empty:
    st.warning("Not enough historical signal to compute trends. Try uploading a CSV with a wider post_date range.")
    st.stop()

declining = summary_df[summary_df["category"] == "Declining"].sort_values(
    by=["p_value", "last_mentions"], ascending=[True, False]
)
emerging = summary_df[summary_df["category"] == "Emerging"].sort_values(
    by=["p_value", "last_mentions"], ascending=[True, False]
)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 Declining skills</div>', unsafe_allow_html=True)
    if declining.empty:
        st.info("No statistically significant declines found for the selected settings.")
    else:
        show_cols = [
            "skill",
            "total_mentions",
            "first_mentions",
            "last_mentions",
            "slope_mentions_per_step",
            "p_value",
            "estimated_months_to_fade",
        ]
        st.dataframe(declining[show_cols].head(12), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌱 Emerging skills</div>', unsafe_allow_html=True)
    if emerging.empty:
        st.info("No statistically significant emergence found for the selected settings.")
    else:
        show_cols = [
            "skill",
            "total_mentions",
            "first_mentions",
            "last_mentions",
            "slope_mentions_per_step",
            "p_value",
            "estimated_months_to_emerge",
        ]
        st.dataframe(emerging[show_cols].head(12), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

skill_options = (
    summary_df.sort_values("total_mentions", ascending=False)["skill"].head(30).tolist()
)
selected = st.selectbox("Timeline for a skill", options=skill_options, index=0)

if selected:
    tdf = (
        pivot.reset_index()
        .rename(columns={"bucket": "bucket"})
        [["bucket", selected]]
        .rename(columns={selected: "mentions"})
    )
    fig = px.line(tdf, x="bucket", y="mentions", markers=True)
    fig.update_layout(**plotly_dark_layout(height=380))
    fig.update_xaxes(title_text="Time bucket")
    fig.update_yaxes(title_text=f"Mentions ({freq == 'M' and 'per month' or 'per week'})")
    st.plotly_chart(fig, use_container_width=True)

