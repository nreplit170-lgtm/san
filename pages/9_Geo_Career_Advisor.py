"""
Page 9 — Geo-Aware Career Advisor (Feature 3)

Folium map, location quotients, relocation ranking from posting data,
and ML risk delta by location tier (same model as Job Risk page).
"""
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.geo_career_advisor import (
    aggregate_city_labour_market,
    build_folium_map,
    extract_user_skill_phrases,
    load_city_reference,
    normalize_city_key,
    rank_relocation_targets,
    relocation_model_delta_pct,
    resolve_city_row,
    skill_location_quotients,
    skill_match_rate_in_subset,
)
from src.job_market_pulse import load_job_postings, postings_with_city_key
from src.job_risk_model import EDUCATION_LEVELS, INDUSTRY_GROWTH
from src.ui_helpers import DARK_CSS

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
        Map hiring intensity by city, measure skill demand with location quotients,
        rank relocation targets from your data, and see modeled risk change by location tier.
    </div>
</div>""", unsafe_allow_html=True)

df_jobs = load_job_postings()
if df_jobs.empty:
    st.error("No job postings loaded. Add `data/market_pulse/job_postings_sample.csv` or upload from Market Pulse workflow.")
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

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Hiring intensity map</div>', unsafe_allow_html=True)
st.caption(
    "Basemap: CartoDB Positron (OSM). Circle area scales with posting count in your file; heat layer is optional density."
)
m = build_folium_map(agg, highlight_city_key=user_ck, extra_marker=extra_pin)
st_folium(m, width=None, height=480, returned_objects=[])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Relocation ranking", "Location quotients", "Modeled risk by tier"])

with tab1:
    st.markdown(
        "Ranks cities using a **transparent composite**: "
        "55% normalized posting volume vs your city, 45% share of local jobs matching your skills."
    )
    rk = rank_relocation_targets(df_jobs, user_ck, phrases)
    if not rk.empty:
        st.dataframe(
            rk.rename(
                columns={
                    "volume_vs_yours": "Posting volume vs yours (×)",
                    "your_skill_match_rate": "Skill match rate (local)",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("Not enough city-level rows to rank.")

with tab2:
    st.markdown(
        "**Location quotient (LQ)** ≈ (local mention rate) ÷ (national mention rate). "
        "LQ > 1 means the skill appears more often in that city than in the full sample."
    )
    if not phrases:
        st.info("Enter skills to compute LQs for your city.")
    else:
        lq = skill_location_quotients(df_jobs, user_ck, phrases)
        if lq.empty:
            st.warning("Could not resolve your city in the posting extract for LQ.")
        else:
            st.dataframe(lq, use_container_width=True)
        local_df = dkey[dkey["city_key"] == user_ck]
        if len(local_df) and phrases:
            rate = skill_match_rate_in_subset(local_df, phrases)
            st.metric("Your skill coverage in this city", f"{rate:.1%} of local postings")

with tab3:
    st.markdown(
        "Uses the **same logistic job-risk model** as the Job Risk page: "
        "only `LOCATION_OPTIONS` tier changes between rows — no hand-picked “+40%” claims."
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
    tier_labels = ["Metro / Tier-1 (model)", "Tier-2 city (model)", "Smaller town / rural (model)"]
    st.caption(f"Your city’s reference tier: **{tier_labels[from_tier]}** (from `data/geo/india_city_reference.csv`).")

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
    m3.metric("Δ probability", f"{dpp:+} pp", help="Percentage points; negative = lower modeled risk.")
    if target_city == user_ck:
        st.caption("Pick a different target city to compare tiers.")
