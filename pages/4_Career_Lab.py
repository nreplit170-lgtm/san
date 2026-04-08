"""
Page 4 — Career Lab
Skill demand chart, growth vs risk bubble chart, career path cards.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout, API_BASE_URL

st.set_page_config(page_title="Career Lab | UIP", page_icon="💼", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 💼 Career Lab")
    shock_intensity = st.slider("Shock Intensity", 0.0, 0.6, 0.3, 0.05)
    recovery_rate   = st.slider("Recovery Rate", 0.05, 0.6, 0.3, 0.05)
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/2_Simulator.py", label="🧪 Simulator")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")
    st.page_link("pages/10_Skill_Obsolescence.py", label="⚡ Skill Obsolescence")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">💼 Career Intelligence Lab</div>
    <div class="hero-subtitle">Discover which sectors are growing, which are at risk, and what skills to build</div>
</div>""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_career_data(si, rr):
    try:
        r = requests.post(f"{API_BASE_URL}/simulate",
                          json={"shock_intensity": si, "shock_duration": 2,
                                "recovery_rate": rr, "forecast_horizon": 6},
                          timeout=20)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

data = get_career_data(shock_intensity, recovery_rate)
if not data:
    st.error("⚠️ Cannot connect to API. Start: `uvicorn src.api:app --reload`")
    st.stop()

career = data.get("career_advice", {})
sector_raw = data.get("sector_impact", [])
sector_df = pd.DataFrame(sector_raw) if sector_raw else pd.DataFrame()

growth_sectors  = career.get("growth_sectors", [])
risk_sectors    = career.get("risk_sectors", [])
skills          = career.get("recommended_skills", [])
narrative       = career.get("narrative", "No guidance available.")

# ─── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_kpi_card("🌱", "Growth Sectors", str(len(growth_sectors)), delta_type="down"), unsafe_allow_html=True)
with c2:
    st.markdown(render_kpi_card("⚠️", "Risk Sectors", str(len(risk_sectors)), delta_type="up"), unsafe_allow_html=True)
with c3:
    st.markdown(render_kpi_card("🎓", "Skills to Learn", str(len(skills)), delta_type="neutral"), unsafe_allow_html=True)
with c4:
    ew = data.get("indices", {}).get("early_warning", "🟢 Stable")
    st.markdown(render_kpi_card("🚦", "Market Outlook", ew.split(" ",1)[-1] if " " in ew else ew, delta_type="neutral"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Top row: Sector risk/growth + Skills chart ───────────────────────────────
col_l, col_r = st.columns([1, 1])

with col_l:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Sector Opportunity vs Risk</div>', unsafe_allow_html=True)

    if not sector_df.empty:
        sector_df_plot = sector_df.copy()
        # Normalize stress to 0-1 for plotting purposes
        sector_df_plot["Stress_Norm"] = sector_df_plot["Stress_Score"] / 100.0
        sector_df_plot["Opportunity"] = 1 - sector_df_plot["Stress_Norm"]
        sector_df_plot["Category"] = sector_df_plot.apply(
            lambda r: "🌱 Growth" if r["Resilience_Score"] > 60 and r["Stress_Score"] < 40 else "⚠️ Risk" if r["Stress_Score"] > 60 else "⚖️ Neutral", axis=1
        )
        color_map = {"🌱 Growth": "#10b981", "⚠️ Risk": "#ef4444", "⚖️ Neutral": "#f59e0b"}
        fig_bub = px.scatter(
            sector_df_plot,
            x="Resilience_Score", y="Stress_Score",
            size="Opportunity",
            color="Category",
            text="Sector",
            color_discrete_map=color_map,
            size_max=50,
            range_x=[0, 100], range_y=[0, 100]
        )
        fig_bub.update_traces(
            textfont=dict(color="white", size=10),
            textposition="top center",
            marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)")),
        )
        fig_bub.update_layout(**plotly_dark_layout(height=340, showlegend=True))
        fig_bub.update_xaxes(title_text="Resilience Score", gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748b"))
        fig_bub.update_yaxes(title_text="Stress Score", gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748b"))
        st.plotly_chart(fig_bub, use_container_width=True)
    else:
        st.info("Sector data not available")
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎓 In-Demand Skills Ranking</div>', unsafe_allow_html=True)

    if skills:
        skill_scores = [round(1.0 - i * 0.08, 2) for i in range(len(skills))]
        skill_df = pd.DataFrame({"Skill": skills, "Demand": skill_scores})
        fig_skill = go.Figure(go.Bar(
            x=skill_df["Demand"],
            y=skill_df["Skill"],
            orientation="h",
            marker=dict(
                color=skill_df["Demand"],
                colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#06b6d4"]],
                line=dict(width=0),
            ),
            text=[f"{v:.0%}" for v in skill_df["Demand"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_skill.update_layout(**plotly_dark_layout(height=340, showlegend=False, margin=dict(l=10, r=60, t=10, b=10)))
        fig_skill.update_xaxes(range=[0, 1.2], title_text="Demand Index", showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748b"))
        fig_skill.update_yaxes(title_text="", gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748b"))
        st.plotly_chart(fig_skill, use_container_width=True)
    else:
        st.info("Skills data not available")
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Sector cards: growth vs risk ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_g, col_rsk = st.columns(2)

with col_g:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌱 Growth Sectors</div>', unsafe_allow_html=True)
    if growth_sectors:
        for idx, s in enumerate(growth_sectors):
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.7rem; padding:0.7rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);">
                <div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3);
                            border-radius:50%; width:28px; height:28px; display:flex; align-items:center;
                            justify-content:center; font-size:0.75rem; font-weight:700; color:#10b981;">
                    {idx+1}
                </div>
                <div style="flex:1;">
                    <div style="color:#e2e8f0; font-weight:600; font-size:0.9rem;">{s}</div>
                </div>
                <span class="badge badge-success">Growing</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No growth sectors identified")
    st.markdown("</div>", unsafe_allow_html=True)

with col_rsk:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ At-Risk Sectors</div>', unsafe_allow_html=True)
    if risk_sectors:
        for idx, s in enumerate(risk_sectors):
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.7rem; padding:0.7rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);">
                <div style="background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3);
                            border-radius:50%; width:28px; height:28px; display:flex; align-items:center;
                            justify-content:center; font-size:0.75rem; font-weight:700; color:#ef4444;">
                    {idx+1}
                </div>
                <div style="flex:1;">
                    <div style="color:#e2e8f0; font-weight:600; font-size:0.9rem;">{s}</div>
                </div>
                <span class="badge badge-danger">At Risk</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-risk sectors identified")
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Skills wall + Career narrative ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_sw, col_nar = st.columns([1, 1])

with col_sw:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏅 Your Skill Roadmap</div>', unsafe_allow_html=True)
    if skills:
        chips = "".join([f'<span class="skill-chip">{s}</span>' for s in skills])
        st.markdown(f'<div style="line-height:2.5; padding:0.5rem 0;">{chips}</div>', unsafe_allow_html=True)
    else:
        st.info("No skills data available")
    st.markdown("</div>", unsafe_allow_html=True)

with col_nar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Career Guidance Narrative</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.15);
                border-radius:14px; padding:1.2rem; line-height:1.7;">
        <p style="color:#cbd5e1; font-size:0.92rem; margin:0;">{narrative}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
