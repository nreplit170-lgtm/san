"""
Page 10 — Skill Obsolescence Detector (Feature 9)

Detects declining vs emerging skills from posting-text time-series.
Enhanced: KPI summary, trend scatter, multi-skill timeline, personal skills-at-risk checker, export.
"""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    st.markdown("**🌐 Navigation**")
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

st.markdown("""
<div class="page-hero">
  <div class="hero-title">📉 Skill Obsolescence Detector</div>
  <div class="hero-subtitle">
    Detect declining vs emerging skills using statistical trend analysis on posting text.
    Know what to upskill into — and what to phase out of your CV.
  </div>
</div>
""", unsafe_allow_html=True)

upload = st.file_uploader(
    "Optional: upload historical job postings CSV",
    type=["csv"],
    help="Expected columns: post_date, job_title, description. Other columns are ignored.",
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
    freq = st.selectbox(
        "Trend bucket size", ["M", "W"], index=0,
        format_func=lambda x: "Monthly" if x == "M" else "Weekly",
    )
    top_k = st.slider("Analyse top skills (by total mentions)", 6, 30, 12, 1)
    min_total_mentions = st.slider("Min mentions to consider a skill", 1, 50, 6, 1)
    alpha = st.selectbox("Significance level (alpha)", [0.10, 0.05, 0.01], index=1)
    slope_threshold_log = st.slider(
        "Min trend strength (|slope| on log1p counts)",
        0.005, 0.20, 0.02, 0.005,
        help="Higher = fewer skills flagged; tuned for noisy job text.",
    )
    category_min_change_ratio = st.slider(
        "Min relative change to label Emerging/Declining",
        1.2, 3.5, 1.8, 0.1,
        help="Emerging: last >= first × ratio. Declining: last <= first / ratio.",
    )
    fade_threshold_mentions = st.slider(
        "Fade threshold (mentions per bucket)", 0, 3, 1, 1,
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
    st.warning(
        "Not enough historical signal to compute trends. "
        "Try uploading a CSV with a wider post_date range."
    )
    st.stop()

declining = summary_df[summary_df["category"] == "Declining"].sort_values(
    by=["p_value", "last_mentions"], ascending=[True, False]
)
emerging = summary_df[summary_df["category"] == "Emerging"].sort_values(
    by=["p_value", "last_mentions"], ascending=[True, False]
)
stable = summary_df[summary_df["category"] == "Stable"]

# ── KPI summary ───────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Analysis summary</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Skills analysed", len(summary_df))
k2.metric("🌱 Emerging", len(emerging))
k3.metric("➡ Stable", len(stable))
k4.metric("📉 Declining", len(declining))
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Declining / Emerging tables ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 Declining skills</div>', unsafe_allow_html=True)
    if declining.empty:
        st.info("No statistically significant declines found for the selected settings.")
    else:
        show_cols = [
            "skill", "total_mentions", "first_mentions", "last_mentions",
            "slope_mentions_per_step", "p_value", "estimated_months_to_fade",
        ]
        def _style_declining(val) -> str:
            try:
                if float(val) < 0:
                    return "color: #f87171; font-weight: 700;"
            except (TypeError, ValueError):
                pass
            return ""
        st.dataframe(
            declining[show_cols].head(12).style.map(
                _style_declining, subset=["slope_mentions_per_step"]
            ),
            width='stretch',
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌱 Emerging skills</div>', unsafe_allow_html=True)
    if emerging.empty:
        st.info("No statistically significant emergence found for the selected settings.")
    else:
        show_cols = [
            "skill", "total_mentions", "first_mentions", "last_mentions",
            "slope_mentions_per_step", "p_value", "estimated_months_to_emerge",
        ]
        def _style_emerging(val) -> str:
            try:
                if float(val) > 0:
                    return "color: #34d399; font-weight: 700;"
            except (TypeError, ValueError):
                pass
            return ""
        st.dataframe(
            emerging[show_cols].head(12).style.map(
                _style_emerging, subset=["slope_mentions_per_step"]
            ),
            width='stretch',
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Trend scatter: slope vs total mentions ─────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📊 Trend landscape: slope vs total mentions</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Each bubble is a skill. Position = (total demand, trend slope). "
    "Size = total mentions. Right half = rising demand; left = falling."
)
CATEGORY_COLORS = {"Emerging": "#34d399", "Stable": "#6366f1", "Declining": "#f87171"}
scatter_df = summary_df.copy()
scatter_df["colour"] = scatter_df["category"].map(CATEGORY_COLORS)
fig_scatter = px.scatter(
    scatter_df,
    x="slope_mentions_per_step",
    y="total_mentions",
    color="category",
    size="total_mentions",
    size_max=40,
    text="skill",
    color_discrete_map=CATEGORY_COLORS,
    hover_data={"p_value": True, "first_mentions": True, "last_mentions": True},
)
fig_scatter.add_vline(x=0, line_dash="dot", line_color="#fbbf24",
                      annotation_text="No change", annotation_position="top right")
fig_scatter.update_traces(textposition="top center", textfont_size=10)
fig_scatter.update_layout(
    **plotly_dark_layout(height=420),
    xaxis_title="Slope (mentions per bucket)",
    yaxis_title="Total mentions",
    legend_title="Category",
)
st.plotly_chart(fig_scatter, width='stretch')
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Multi-skill timeline ───────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Skill timeline comparison</div>', unsafe_allow_html=True)
skill_options = (
    summary_df.sort_values("total_mentions", ascending=False)["skill"].head(30).tolist()
)
default_compare = skill_options[:min(4, len(skill_options))]
selected_multi = st.multiselect(
    "Select skills to compare (up to 8)",
    options=skill_options,
    default=default_compare,
    max_selections=8,
    key="obs_multi",
)

if selected_multi:
    avail = [s for s in selected_multi if s in pivot.columns]
    if avail:
        tdf = pivot[avail].reset_index()
        tlong = tdf.melt(id_vars=["bucket"], var_name="skill", value_name="mentions")
        tlong["bucket"] = pd.to_datetime(tlong["bucket"], errors="coerce")
        cat_map = summary_df.set_index("skill")["category"].to_dict()
        tlong["category"] = tlong["skill"].map(cat_map)

        color_seq = [CATEGORY_COLORS.get(cat_map.get(s, "Stable"), "#6366f1") for s in avail]
        fig_multi = px.line(
            tlong,
            x="bucket",
            y="mentions",
            color="skill",
            markers=True,
            color_discrete_sequence=color_seq,
        )
        fig_multi.update_layout(**plotly_dark_layout(height=400))
        fig_multi.update_xaxes(title_text="Time bucket")
        fig_multi.update_yaxes(title_text=f"Mentions ({'per month' if freq == 'M' else 'per week'})")
        st.plotly_chart(fig_multi, width='stretch')
    else:
        st.info("Selected skills not found in trend data.")
else:
    st.info("Select at least one skill above.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Personal skills-at-risk checker ───────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">🎯 Your skills at risk</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Enter your current skill set and see which are declining, stable, or "
    "on the rise — based on this dataset's trend signals."
)
user_skill_input = st.text_input(
    "Your skills (comma-separated)",
    placeholder="e.g. Python, SQL, Hadoop, Tableau, AWS",
    key="obs_user_skills",
)
if user_skill_input.strip():
    user_skills_lower = [s.strip().lower() for s in user_skill_input.split(",") if s.strip()]
    skill_cat_map = summary_df.set_index("skill")["category"].to_dict()
    risk_rows = []
    for us in user_skills_lower:
        matched = None
        for known_skill in skill_cat_map:
            if us in known_skill or known_skill in us:
                matched = known_skill
                break
        if matched:
            row_data = summary_df[summary_df["skill"] == matched].iloc[0]
            risk_rows.append({
                "Your skill": us,
                "Matched to": matched,
                "Category": row_data["category"],
                "Slope": row_data["slope_mentions_per_step"],
                "p-value": row_data["p_value"],
                "Total mentions": row_data["total_mentions"],
            })
        else:
            risk_rows.append({
                "Your skill": us,
                "Matched to": "—",
                "Category": "Not in dataset",
                "Slope": None,
                "p-value": None,
                "Total mentions": None,
            })

    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)

        def _style_cat(val: str) -> str:
            return {
                "Declining": "background-color: rgba(248,113,113,0.12); color: #f87171; font-weight:700;",
                "Emerging": "background-color: rgba(52,211,153,0.12); color: #34d399; font-weight:700;",
                "Stable": "color: #94a3b8;",
                "Not in dataset": "color: #6b7280; font-style: italic;",
            }.get(str(val), "")

        st.dataframe(
            risk_df.style.map(_style_cat, subset=["Category"]),
            width='stretch',
            hide_index=True,
        )

        at_risk = [r["Your skill"] for r in risk_rows if r["Category"] == "Declining"]
        rising = [r["Your skill"] for r in risk_rows if r["Category"] == "Emerging"]
        if at_risk:
            st.markdown(
                "**Skills to watch / phase out:** "
                + " &nbsp;·&nbsp; ".join(
                    f'<span style="background:rgba(248,113,113,0.12);border:1px solid #f87171;'
                    f'border-radius:4px;padding:2px 8px;color:#f87171;">{s}</span>'
                    for s in at_risk
                ),
                unsafe_allow_html=True,
            )
        if rising:
            st.markdown(
                "**Skills to highlight / deepen:** "
                + " &nbsp;·&nbsp; ".join(
                    f'<span style="background:rgba(52,211,153,0.12);border:1px solid #34d399;'
                    f'border-radius:4px;padding:2px 8px;color:#34d399;">{s}</span>'
                    for s in rising
                ),
                unsafe_allow_html=True,
            )
else:
    st.info("Enter your skills above to check your personal obsolescence exposure.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📥 Export results</div>', unsafe_allow_html=True)
export_parts = ["=== SKILL OBSOLESCENCE SUMMARY ===\n", summary_df.to_csv(index=False)]
if not pivot.empty:
    export_parts += ["\n\n=== RAW MENTION PIVOT ===\n", pivot.to_csv()]
csv_bytes = "".join(export_parts).encode()
st.download_button(
    label="⬇ Download full results (CSV)",
    data=csv_bytes,
    file_name="skill_obsolescence_results.csv",
    mime="text/csv",
)
st.markdown("</div>", unsafe_allow_html=True)
