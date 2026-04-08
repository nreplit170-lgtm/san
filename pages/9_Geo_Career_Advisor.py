"""
Page 9 — Geo-Aware Career Advisor (Feature 8)

Folium map, city posting volume chart, location quotients with bar chart,
relocation ranking with colour-coded skill fit, and ML risk comparison
across all tiers for the current profile.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src.geo_career_advisor import (
    aggregate_city_labour_market,
    build_folium_map,
    extract_user_skill_phrases,
    load_city_reference,
    normalize_city_key,
    postings_with_city_key,
    rank_relocation_targets,
    relocation_model_delta_pct,
    resolve_city_row,
    skill_location_quotients,
    skill_match_rate_in_subset,
)
from src.job_market_pulse import load_job_postings
from src.job_risk_model import EDUCATION_LEVELS, INDUSTRY_GROWTH, LOCATION_OPTIONS, predict_job_risk
from src.ui_helpers import DARK_CSS, plotly_dark_layout

st.set_page_config(page_title="Geo Career | UIP", page_icon="🗺️", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🗺️ Geo Career Advisor")
    st.caption(
        "WGS84 + Folium + optional Nominatim (OSM). "
        "Metrics are computed from your posting CSV — swap in real feeds for production."
    )
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
    st.page_link("pages/10_Skill_Obsolescence.py", label="⚡ Skill Obsolescence")


@st.cache_data(ttl=86400, show_spinner=False)
def cached_geocode(query: str):
    from src.geo_career_advisor import geocode_place
    return geocode_place(query)


st.markdown("""
<div class="page-hero">
    <div class="hero-title">🗺️ Geo-Aware Career Advisor</div>
    <div class="hero-subtitle">
        Map hiring intensity by city, compare skill demand with location quotients,
        rank relocation targets, and model your risk change across location tiers.
    </div>
</div>""", unsafe_allow_html=True)

df_jobs = load_job_postings()
if df_jobs.empty:
    st.error("No job postings loaded. Add `data/market_pulse/job_postings_sample.csv` or upload from Market Pulse page.")
    st.stop()

agg = aggregate_city_labour_market(df_jobs)
dkey = postings_with_city_key(df_jobs)

loc_values = sorted(dkey["location"].dropna().astype(str).unique().tolist())
city_keys_in_data = sorted(dkey["city_key"].unique().tolist())

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    home_display = st.selectbox("Your city (from dataset)", loc_values, index=0)
with c2:
    skills = st.text_area(
        "Your skills (comma-separated)",
        placeholder="python, sql, aws",
        height=68,
        help="Used for skill match rate and location quotients.",
    )
with c3:
    geocode_query = st.text_input(
        "Optional: geocode another place (Nominatim)",
        placeholder="e.g. Indore",
        help="Respects OpenStreetMap usage policy; results cached 24h. Requires network.",
    )

phrases = extract_user_skill_phrases(skills)
user_ck = normalize_city_key(home_display)
extra_pin = None
if geocode_query.strip():
    geo = cached_geocode(geocode_query.strip())
    if geo:
        extra_pin = (geo[0], geo[1], geo[2])
    else:
        st.caption("Geocoder returned no result — check spelling or try a larger nearby city.")

# ── Hiring intensity map ───────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Hiring intensity map</div>', unsafe_allow_html=True)
st.caption(
    "Basemap: CartoDB Positron (OSM). Circle area scales with posting count; heat layer shows density."
)
m = build_folium_map(agg, highlight_city_key=user_ck, extra_marker=extra_pin)
st_folium(m, width=None, height=480, returned_objects=[])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── City posting volume + salary chart ────────────────────────────────────────
if not agg.empty:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📊 City hiring volume & median salary</div>',
        unsafe_allow_html=True,
    )
    agg_disp = agg.copy()
    agg_disp["City"] = agg_disp.get("display_name", agg_disp["city_key"])
    agg_disp = agg_disp.sort_values("postings", ascending=False).head(15)

    fig_city = go.Figure()
    fig_city.add_trace(go.Bar(
        x=agg_disp["City"],
        y=agg_disp["postings"],
        name="Postings",
        marker_color=[
            "#06b6d4" if ck == user_ck else "#6366f1"
            for ck in agg_disp["city_key"]
        ],
        text=agg_disp["postings"],
        textposition="outside",
    ))
    if "median_lpa" in agg_disp.columns and agg_disp["median_lpa"].notna().any():
        fig_city.add_trace(go.Scatter(
            x=agg_disp["City"],
            y=agg_disp["median_lpa"],
            name="Median salary (LPA)",
            mode="lines+markers",
            marker=dict(color="#34d399", size=8),
            line=dict(color="#34d399", width=2),
            yaxis="y2",
        ))
        fig_city.update_layout(
            yaxis2=dict(
                title="Median LPA",
                overlaying="y",
                side="right",
                showgrid=False,
                color="#34d399",
            )
        )
    fig_city.update_layout(
        **plotly_dark_layout(height=380),
        xaxis_title="City",
        yaxis_title="Job postings",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        barmode="group",
    )
    st.plotly_chart(fig_city, use_container_width=True)
    st.caption("Cyan bar = your selected city.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Relocation ranking", "Location quotients", "Modeled risk by tier"])

with tab1:
    st.markdown(
        "Ranks cities by a **transparent composite**: "
        "55% normalized posting volume vs your city, 45% share of local jobs matching your skills."
    )
    rk = rank_relocation_targets(df_jobs, user_ck, phrases)
    if rk.empty:
        st.info("Not enough city-level rows to rank.")
    else:
        rk_disp = rk.rename(columns={
            "display_name": "City",
            "postings": "Postings",
            "volume_vs_yours": "Volume vs yours (×)",
            "your_skill_match_rate": "Skill match rate",
            "score": "Composite score",
        })[["City", "Postings", "Volume vs yours (×)", "Skill match rate", "Composite score"]]

        def _style_skill_fit(val) -> str:
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v >= 0.5:
                return "background-color: rgba(16,185,129,0.15); color: #34d399; font-weight:700;"
            if v >= 0.25:
                return "color: #fbbf24;"
            return "color: #f87171;"

        def _style_score(val) -> str:
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v >= 0.5:
                return "color: #34d399; font-weight: 700;"
            if v >= 0.3:
                return "color: #6366f1;"
            return ""

        st.dataframe(
            rk_disp.style
                .map(_style_skill_fit, subset=["Skill match rate"])
                .map(_style_score, subset=["Composite score"]),
            use_container_width=True,
            hide_index=True,
        )

        csv_rk = rk_disp.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export relocation ranking (CSV)",
            data=csv_rk,
            file_name="relocation_ranking.csv",
            mime="text/csv",
        )

        top5 = rk_disp.head(5)
        fig_rk = px.bar(
            top5,
            x="Composite score",
            y="City",
            orientation="h",
            color="Skill match rate",
            color_continuous_scale=["#f87171", "#fbbf24", "#34d399"],
            title="Top 5 relocation targets",
        )
        fig_rk.update_layout(**plotly_dark_layout(height=280))
        fig_rk.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_rk, use_container_width=True)

with tab2:
    st.markdown(
        "**Location quotient (LQ)** ≈ (local mention rate) ÷ (national mention rate). "
        "LQ > 1 means the skill appears more often in that city than in the full sample."
    )
    if not phrases:
        st.info("Enter your skills above to compute LQs for your city.")
    else:
        lq = skill_location_quotients(df_jobs, user_ck, phrases)
        if lq.empty:
            st.warning("Could not resolve your city in the posting extract for LQ — try a city from the dataset list.")
        else:
            col_lq1, col_lq2 = st.columns([2, 3])
            with col_lq1:
                st.dataframe(lq, use_container_width=True)
            with col_lq2:
                lq_plot = lq.dropna(subset=["lq"]).copy()
                lq_plot["lq"] = lq_plot["lq"].astype(float)
                lq_plot["above"] = lq_plot["lq"] >= 1.0
                fig_lq = px.bar(
                    lq_plot,
                    x="lq",
                    y="skill",
                    orientation="h",
                    color="above",
                    color_discrete_map={True: "#34d399", False: "#f87171"},
                    title=f"Location quotients — {home_display}",
                )
                fig_lq.add_vline(x=1.0, line_dash="dot", line_color="#fbbf24",
                                 annotation_text="National avg", annotation_position="top right")
                fig_lq.update_layout(**plotly_dark_layout(height=max(240, 30 * len(lq_plot))))
                fig_lq.update_yaxes(autorange="reversed")
                fig_lq.update_layout(showlegend=False)
                st.plotly_chart(fig_lq, use_container_width=True)

        local_df = dkey[dkey["city_key"] == user_ck]
        if len(local_df) and phrases:
            rate = skill_match_rate_in_subset(local_df, phrases)
            st.metric(
                "Your skill coverage in this city",
                f"{rate:.1%} of local postings",
                help="Share of local job postings that mention at least one of your skills.",
            )

with tab3:
    st.markdown(
        "Uses the **same logistic job-risk model** as the Job Risk page: "
        "only the location tier changes — no hard-coded percentage claims."
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        education = st.selectbox("Education", EDUCATION_LEVELS, index=2, key="geo_edu")
    with col_b:
        experience = st.slider("Years experience", 0, 40, 3, key="geo_exp")
    with col_c:
        industry = st.selectbox("Industry", list(INDUSTRY_GROWTH.keys()), key="geo_ind")

    row = resolve_city_row(home_display)
    from_tier = int(row["market_tier_index"]) if row is not None else 1
    tier_labels = ["Metro / Tier-1", "Tier-2 city", "Smaller town / rural"]
    st.caption(f"Your city's reference tier: **{tier_labels[from_tier]}** (from `data/geo/india_city_reference.csv`).")

    target_opts = [r for r in city_keys_in_data if r != user_ck]
    if not target_opts:
        target_opts = city_keys_in_data
    ref_df = load_city_reference()
    labels = {
        r["city_key"]: str(r["display_name"])
        for _, r in ref_df.iterrows()
    } if not ref_df.empty else {}
    target_city = st.selectbox(
        "Compare modeled risk if you were based in",
        target_opts,
        format_func=lambda k: labels.get(k, k.replace("_", " ").title()),
    )
    match = ref_df[ref_df["city_key"] == target_city] if not ref_df.empty else pd.DataFrame()
    to_tier = int(match.iloc[0]["market_tier_index"]) if len(match) else 1

    p0, p1, dpp = relocation_model_delta_pct(
        skills, education, experience, industry, from_tier, to_tier
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk % (current tier)", f"{p0}%")
    m2.metric("Risk % (target tier)", f"{p1}%")
    delta_color = "normal" if dpp <= 0 else "inverse"
    m3.metric("Δ probability", f"{dpp:+} pp", delta=f"{dpp:+} pp", delta_color=delta_color,
              help="Percentage points; negative = lower modeled risk.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk across all 3 tiers
    all_tier_risks = []
    for ti, tlabel in enumerate(tier_labels):
        loc = LOCATION_OPTIONS[ti]
        r = predict_job_risk(skills, education, experience, loc, industry)
        all_tier_risks.append({
            "Tier": tlabel,
            "Risk %": r.high_risk_probability_pct,
            "is_current": ti == from_tier,
        })
    tier_df = pd.DataFrame(all_tier_risks)

    fig_tier = go.Figure(go.Bar(
        x=tier_df["Tier"],
        y=tier_df["Risk %"],
        marker_color=[
            "#06b6d4" if row["is_current"] else "#6366f1"
            for _, row in tier_df.iterrows()
        ],
        text=[f"{v}%" for v in tier_df["Risk %"]],
        textposition="outside",
    ))
    fig_tier.update_layout(
        **plotly_dark_layout(height=320),
        title="Modeled risk across all location tiers (same profile)",
        yaxis_title="High-risk probability (%)",
        xaxis_title="Location tier",
    )
    st.plotly_chart(fig_tier, use_container_width=True)
    st.caption("Cyan bar = your current city's tier.")

    if target_city == user_ck:
        st.caption("Pick a different target city to compare a different tier.")
