"""
Page 8 — Job Market Pulse (Feature 2)

Skill/role demand from bundled CSV (or your upload). Weekly trends + salary bands.
Standalone — does not use the FastAPI backend.
"""
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.job_market_pulse import (
    default_jobs_csv_path,
    jobs_from_upload,
    load_job_postings,
    role_demand_counts,
    salary_summary_by_role,
    skill_demand_counts,
    weekly_skill_trends,
)
from src.ui_helpers import DARK_CSS, plotly_dark_layout

st.set_page_config(page_title="Market Pulse | UIP", page_icon="📡", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📡 Job Market Pulse")
    st.caption("Swap in a Kaggle or ATS export via upload — same columns as the sample.")
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/2_Simulator.py", label="🧪 Simulator")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")
    st.page_link("pages/10_Skill_Obsolescence.py", label="⚡ Skill Obsolescence")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">📡 Job Market Pulse</div>
    <div class="hero-subtitle">
        Surface in-demand skills and role families from job text — frequency, weekly momentum, and salary midpoints.
    </div>
</div>""", unsafe_allow_html=True)

upload = st.file_uploader(
    "Optional: CSV upload",
    type=["csv"],
    help="Expected columns: post_date, job_title, description, location, salary_min_lpa, salary_max_lpa (extras ignored).",
)

if upload is not None:
    df = jobs_from_upload(io.BytesIO(upload.getvalue()))
else:
    df = load_job_postings()

if df.empty:
    st.error(
        "No job data found. Add `data/market_pulse/job_postings_sample.csv` or upload a CSV."
    )
    st.caption(f"Default path: `{default_jobs_csv_path()}`")
    st.stop()

loc_opts = ["All locations"]
if "location" in df.columns:
    loc_opts += sorted(df["location"].dropna().astype(str).unique().tolist())

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    loc = st.selectbox("Filter by location", loc_opts)
with c2:
    top_n = st.slider("Top skills to show", 5, 25, 12)
with c3:
    trend_skills = st.slider("Skills in trend chart", 3, 8, 5)

filtered = df if loc == "All locations" else df[df["location"].astype(str) == loc]

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dataset snapshot</div>', unsafe_allow_html=True)
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Postings", len(filtered))
with k2:
    dmin = filtered["post_date"].min() if "post_date" in filtered.columns else pd.NaT
    dmax = filtered["post_date"].max() if "post_date" in filtered.columns else pd.NaT
    span = (
        f"{pd.Timestamp(dmin).date()} → {pd.Timestamp(dmax).date()}"
        if pd.notna(dmin) and pd.notna(dmax)
        else "—"
    )
    st.metric("Date span", span)
with k3:
    titles = filtered.get("job_title", pd.Series(dtype=str))
    st.metric("Unique titles", titles.nunique())
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

skills = skill_demand_counts(filtered).head(top_n)
roles = role_demand_counts(filtered).head(12)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top skills (listing mentions)</div>', unsafe_allow_html=True)
    if skills.empty:
        st.info("No skill phrases matched. Check descriptions or extend lexicon in `job_market_pulse.py`.")
    else:
        sf = skills.reset_index()
        sf.columns = ["skill", "count"]
        fig_s = px.bar(
            sf,
            x="count",
            y="skill",
            orientation="h",
            color="count",
            color_continuous_scale=["#1e1b4b", "#6366f1", "#06b6d4"],
        )
        fig_s.update_layout(**plotly_dark_layout(height=max(320, 24 * len(sf))))
        fig_s.update_yaxes(autorange="reversed")
        fig_s.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_s, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top job role families</div>', unsafe_allow_html=True)
    if roles.empty:
        st.info("No roles to chart.")
    else:
        rf = roles.reset_index()
        rf.columns = ["role", "count"]
        fig_r = px.bar(
            rf,
            x="count",
            y="role",
            orientation="h",
            color="count",
            color_continuous_scale=["#312e81", "#8b5cf6", "#34d399"],
        )
        fig_r.update_layout(**plotly_dark_layout(height=max(320, 22 * len(rf))))
        fig_r.update_yaxes(autorange="reversed")
        fig_r.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_r, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Weekly demand trend (top skills)</div>', unsafe_allow_html=True)
trend_df = weekly_skill_trends(filtered, top_n_skills=trend_skills)
if trend_df.empty:
    st.info("Need valid `post_date` values for weekly trends.")
else:
    twide = trend_df.reset_index()
    tlong = twide.melt(id_vars=["week"], var_name="skill", value_name="mentions")
    tlong["week"] = pd.to_datetime(tlong["week"])
    fig_t = px.line(
        tlong,
        x="week",
        y="mentions",
        color="skill",
        markers=True,
    )
    fig_t.update_layout(**plotly_dark_layout(height=380))
    fig_t.update_xaxes(title_text="Week (start)")
    fig_t.update_yaxes(title_text="Postings mentioning skill")
    st.plotly_chart(fig_t, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Salary insight (median LPA by role)</div>', unsafe_allow_html=True)
sal = salary_summary_by_role(filtered)
if sal.empty:
    st.caption("Add numeric `salary_min_lpa` / `salary_max_lpa` on more rows to populate this.")
else:
    fig_h = go.Figure(
        go.Bar(
            x=sal["median_lpa"],
            y=sal.index.astype(str),
            orientation="h",
            marker=dict(color="rgba(99,102,241,0.85)"),
        )
    )
    fig_h.update_layout(**plotly_dark_layout(height=max(280, 28 * len(sal))))
    fig_h.update_yaxes(autorange="reversed", title_text="")
    fig_h.update_xaxes(title_text="Median of (min+max)/2 LPA")
    st.plotly_chart(fig_h, use_container_width=True)
    st.dataframe(sal, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
