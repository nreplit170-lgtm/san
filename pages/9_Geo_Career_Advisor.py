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
from src.live_data import fetch_labor_market_pulse, get_state_unemployment
from src.live_insights import generate_labor_market_insights
from src.ui_helpers import DARK_CSS, plotly_dark_layout, render_kpi_card

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
tab1, tab2, tab3, tab4 = st.tabs([
    "Relocation ranking", "Location quotients",
    "Modeled risk by tier", "🌐 Live India Context",
])

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

# ── TAB 4 — LIVE INDIA CONTEXT ─────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div style="background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.25);
                border-radius:14px; padding:1rem 1.5rem; margin-bottom:1.5rem;
                display:flex; gap:0.75rem; align-items:flex-start;">
        <div style="font-size:1.3rem;">🌐</div>
        <div>
            <div style="font-size:0.82rem; font-weight:700; color:#818cf8;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">
                Real India Labor Data</div>
            <div style="font-size:0.85rem; color:#94a3b8; line-height:1.55;">
                <strong style="color:#e2e8f0;">National trends</strong> are fetched live from the
                <strong style="color:#e2e8f0;">World Bank Open API</strong>.
                <strong style="color:#e2e8f0;">State-level data</strong> is from the official
                <strong style="color:#e2e8f0;">PLFS 2022-23 report</strong> (MOSPI, Govt. of India) —
                state unemployment is not available on the World Bank API.
                Use this view to ground your city decision in real macro and regional data.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── National KPIs from World Bank
    st.markdown('<div class="section-title">📊 India National Labor Snapshot (World Bank)</div>',
                unsafe_allow_html=True)

    with st.spinner("Fetching live data from World Bank…"):
        wb_data = fetch_labor_market_pulse("India")

    # ── AI Insight Box ─────────────────────────────────────────────────────────
    geo_insights = generate_labor_market_insights(wb_data)
    if geo_insights:
        bullets_html = "".join(
            f'<li style="margin-bottom:0.45rem; color:#cbd5e1; font-size:0.9rem; line-height:1.6;">'
            + s.replace("**", "<strong style='color:#e2e8f0;'>", 1).replace("**", "</strong>", 1)
            + "</li>"
            for s in geo_insights
        )
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.25);
                    border-radius:14px; padding:1rem 1.5rem; margin-bottom:1.4rem;">
            <div style="display:flex; gap:0.6rem; align-items:center; margin-bottom:0.6rem;">
                <span style="font-size:1.1rem;">💡</span>
                <span style="font-size:0.78rem; font-weight:700; color:#818cf8;
                              text-transform:uppercase; letter-spacing:1px;">
                    India Labor Market Context — What This Means for Your City Choice
                </span>
            </div>
            <ul style="margin:0; padding-left:1.2rem;">{bullets_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    KEY_KPIS = [
        ("Unemployment Rate (%)",       "📊", "neutral"),
        ("Youth Unemployment 15-24 (%)","👶", "up"),
        ("Labor Force Participation (%)","💪", "neutral"),
        ("Employment-to-Population (%)","🏭", "neutral"),
    ]
    kpi_cols = st.columns(len(KEY_KPIS))
    for col, (label, icon, dt) in zip(kpi_cols, KEY_KPIS):
        with col:
            series = wb_data.get(label)
            if series is not None and not series.empty:
                latest_val  = series.iloc[-1]["Value"]
                latest_year = int(series.iloc[-1]["Year"])
                delta_txt   = ""
                if len(series) >= 2:
                    prev = series.iloc[-2]["Value"]
                    chg = round(latest_val - prev, 2)
                    arrow = "▲" if chg > 0 else "▼"
                    delta_txt = f"{arrow} {abs(chg)}pp vs {latest_year - 1}"
                st.markdown(
                    render_kpi_card(icon, label, f"{latest_val:.1f}%", delta_txt, dt),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(render_kpi_card(icon, label, "N/A", "Unavailable", "neutral"),
                            unsafe_allow_html=True)

    # ── National unemployment trend sparkline
    if wb_data.get("Unemployment Rate (%)") is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 India Unemployment Trend (World Bank, 1991–2023)</div>',
                    unsafe_allow_html=True)
        ue_series = wb_data["Unemployment Rate (%)"]
        fig_ue = go.Figure()
        fig_ue.add_trace(go.Scatter(
            x=ue_series["Year"], y=ue_series["Value"],
            mode="lines+markers", name="Unemployment Rate",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=5, color="#6366f1"),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            hovertemplate="<b>Year: %{x}</b><br>Rate: %{y:.2f}%<extra></extra>",
        ))
        youth = wb_data.get("Youth Unemployment 15-24 (%)")
        if youth is not None and not youth.empty:
            fig_ue.add_trace(go.Scatter(
                x=youth["Year"], y=youth["Value"],
                mode="lines", name="Youth Unemployment (15-24)",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                hovertemplate="<b>Year: %{x}</b><br>Youth: %{y:.2f}%<extra></extra>",
            ))
        fig_ue.update_layout(
            **plotly_dark_layout(height=360),
            xaxis_title="Year", yaxis_title="Unemployment Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0.3)", font=dict(color="#cbd5e1")),
        )
        st.plotly_chart(fig_ue, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── State-level unemployment breakdown (PLFS 2022-23)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺️ State-Level Unemployment — PLFS 2022-23 (UPS, 15+ yrs)</div>',
                unsafe_allow_html=True)

    state_df = get_state_unemployment()

    view_type = st.radio("View", ["Combined", "Urban vs Rural comparison"],
                         horizontal=True, key="state_view_type")

    if view_type == "Combined":
        color_discrete = dict(
            North="#6366f1", South="#10b981", East="#f59e0b",
            West="#06b6d4", Central="#8b5cf6", Northeast="#ec4899",
        )
        fig_state = px.bar(
            state_df.sort_values("Combined_UE", ascending=True),
            x="Combined_UE", y="State", orientation="h",
            color="Region",
            color_discrete_map=color_discrete,
            text="Combined_UE",
            labels={"Combined_UE": "Unemployment Rate (%)"},
        )
        fig_state.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        india_avg = state_df["Combined_UE"].mean()
        fig_state.add_vline(x=india_avg, line_dash="dot", line_color="#94a3b8",
                            annotation_text=f"India avg ~{india_avg:.1f}%",
                            annotation_position="top right",
                            annotation_font=dict(color="#94a3b8", size=10))
        fig_state.update_layout(
            **plotly_dark_layout(height=max(560, 20 * len(state_df))),
            xaxis_title="Unemployment Rate (%)", yaxis_title="",
        )
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        state_plot = state_df.dropna(subset=["Urban_UE"]).sort_values("Combined_UE", ascending=False).head(20)
        fig_urvr = go.Figure()
        fig_urvr.add_trace(go.Bar(
            x=state_plot["State"], y=state_plot["Urban_UE"],
            name="Urban", marker_color="#6366f1",
        ))
        fig_urvr.add_trace(go.Bar(
            x=state_plot["State"], y=state_plot["Rural_UE"],
            name="Rural", marker_color="#10b981",
        ))
        fig_urvr.update_layout(
            **plotly_dark_layout(height=420),
            barmode="group",
            xaxis_title="State", yaxis_title="Unemployment Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0.3)", font=dict(color="#cbd5e1")),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_urvr, use_container_width=True)

    st.caption("Source: PLFS Annual Report 2022-23, MOSPI, Government of India | UPS = Usual Principal Status")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Region comparison
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏙️ Average Unemployment by Region</div>',
                unsafe_allow_html=True)
    region_avg = (
        state_df.groupby("Region")["Combined_UE"]
        .mean().round(2)
        .reset_index()
        .rename(columns={"Combined_UE": "Avg Unemployment (%)"})
        .sort_values("Avg Unemployment (%)", ascending=False)
    )
    fig_reg = px.bar(
        region_avg, x="Region", y="Avg Unemployment (%)",
        color="Avg Unemployment (%)",
        color_continuous_scale=["#0d9488", "#6366f1", "#ef4444"],
        text="Avg Unemployment (%)",
    )
    fig_reg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_reg.update_layout(
        **plotly_dark_layout(height=320),
        xaxis_title="Region", yaxis_title="Average Unemployment (%)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Export
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Export State Data</div>', unsafe_allow_html=True)
    csv_bytes = state_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download PLFS State Unemployment Data (CSV)",
        csv_bytes,
        file_name="india_state_unemployment_plfs2023.csv",
        mime="text/csv",
    )
    st.caption("Source: PLFS Annual Report 2022-23 | Ministry of Statistics & Programme Implementation (MOSPI)")
