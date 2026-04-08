"""
Page 5 — AI Insights
AI narrative panel, story timeline, macro/sector/recovery cards.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout, API_BASE_URL

st.set_page_config(page_title="AI Insights | UIP", page_icon="🤖", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 AI Insights")
    shock_intensity = st.slider("Shock Intensity", 0.0, 0.6, 0.3, 0.05)
    shock_duration  = st.slider("Shock Duration (yrs)", 0, 5, 2)
    recovery_rate   = st.slider("Recovery Rate", 0.05, 0.6, 0.3, 0.05)
    policy          = st.selectbox("Policy", ["None","Fiscal Stimulus","Monetary Policy","Labor Reforms","Industry Support"])
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/2_Simulator.py", label="🧪 Simulator")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">🤖 AI Intelligence Engine</div>
    <div class="hero-subtitle">Machine-crafted economic narratives, year-by-year story mode, and multi-dimensional insights</div>
</div>""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_insights_data(si, sd, rr, pol):
    try:
        r = requests.post(f"{API_BASE_URL}/simulate",
                          json={"shock_intensity": si, "shock_duration": sd,
                                "recovery_rate": rr, "forecast_horizon": 7,
                                "policy_name": pol if pol != "None" else None},
                          timeout=20)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

data = get_insights_data(shock_intensity, shock_duration, recovery_rate, policy)
if not data:
    st.error("⚠️ Cannot connect to API. Start: `uvicorn src.api:app --reload`")
    st.stop()

insights  = data.get("ai_insights", {})
story     = data.get("story", [])
indices   = data.get("indices", {})
scen_df   = pd.DataFrame(data.get("scenario", []))

# ─── KPI Row ──────────────────────────────────────────────────────────────────
rqi  = indices.get("rqi_label", "N/A")
usi  = indices.get("unemployment_stress_index", "N/A")
ew   = indices.get("early_warning", "N/A")
peak = round(scen_df["Scenario_Unemployment"].max(), 2) if not scen_df.empty else "N/A"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_kpi_card("🧠", "AI Risk Signal", ew.split(" ",1)[-1] if " " in str(ew) else str(ew), delta_type="neutral"), unsafe_allow_html=True)
with c2:
    st.markdown(render_kpi_card("📊", "Stress Index", str(usi), delta_type="neutral"), unsafe_allow_html=True)
with c3:
    st.markdown(render_kpi_card("🔄", "Recovery Quality", str(rqi), delta_type="neutral"), unsafe_allow_html=True)
with c4:
    st.markdown(render_kpi_card("🎯", "Scenario Peak", f"{peak}%", delta_type="neutral"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── AI Summary Banner ────────────────────────────────────────────────────────
summary = insights.get("summary", "")
if summary:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
                border:1px solid rgba(99,102,241,0.25); border-radius:20px; padding:2rem;
                margin-bottom:2rem; position:relative; overflow:hidden;">
        <div style="position:absolute; top:-20px; right:-20px; width:120px; height:120px;
                    background:radial-gradient(circle, rgba(99,102,241,0.2), transparent 70%);
                    border-radius:50%;"></div>
        <div style="display:flex; gap:1rem; align-items:flex-start;">
            <div style="font-size:2rem; flex-shrink:0;">🤖</div>
            <div>
                <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:1.5px; color:#818cf8; margin-bottom:0.5rem;">AI ECONOMIC BRIEF</div>
                <p style="color:#cbd5e1; font-size:1rem; line-height:1.7; margin:0;">{summary}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Insight Trio ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
insight_cards = [
    ("🌍", "Macro View", "macro_insight", "#818cf8"),
    ("🏭", "Sector View", "sector_insight", "#06b6d4"),
    ("🔄", "Recovery Outlook", "recovery_insight", "#10b981"),
]
for col, (icon, title, key, color) in zip([col1, col2, col3], insight_cards):
    text = insights.get(key, "No insight available.")
    with col:
        st.markdown(f"""
        <div class="glass-card" style="height:220px; overflow:auto;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="font-size:0.95rem; font-weight:700; color:#e2e8f0;">{title}</div>
            </div>
            <div style="width:40px; height:3px; background:{color}; border-radius:2px; margin-bottom:0.8rem;"></div>
            <p style="color:#94a3b8; font-size:0.88rem; line-height:1.6; margin:0;">{text}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Story Timeline + Mini Chart ──────────────────────────────────────────────
col_tl, col_chart = st.columns([1, 1])

with col_tl:
    st.markdown('<div class="glass-card" style="max-height:500px; overflow-y:auto;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📖 Year-by-Year Story</div>', unsafe_allow_html=True)
    if story:
        for event in story:
            yr   = event.get("year", "?")
            val  = event.get("value", "?")
            desc = event.get("description", "")
            icon = event.get("icon", "📅")
            phase = event.get("phase", "")
            color = "#ef4444" if "Shock" in phase else "#f59e0b" if "Recovery" in phase else "#10b981"
            st.markdown(f"""
            <div class="timeline-item">
                <div>
                    <div style="width:36px; height:36px; border-radius:50%;
                                background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25);
                                display:flex; align-items:center; justify-content:center; font-size:1rem;">
                        {icon}
                    </div>
                </div>
                <div class="timeline-content">
                    <div style="display:flex; gap:0.5rem; align-items:center; margin-bottom:0.2rem;">
                        <span class="timeline-year">{yr}</span>
                        <span style="font-size:0.78rem; font-weight:700; color:{color};">{val}%</span>
                        {f'<span class="badge badge-info" style="font-size:0.7rem; padding:0.1rem 0.5rem;">{phase}</span>' if phase else ''}
                    </div>
                    <div class="timeline-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No story data available")
    st.markdown("</div>", unsafe_allow_html=True)

with col_chart:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Scenario Trajectory</div>', unsafe_allow_html=True)

    if not scen_df.empty and "Scenario_Unemployment" in scen_df.columns:
        base_df = pd.DataFrame(data.get("baseline", []))

        fig = go.Figure()
        if not base_df.empty:
            fig.add_trace(go.Scatter(
                x=base_df["Year"], y=base_df["Predicted_Unemployment"],
                mode="lines", name="Baseline",
                line=dict(color="#64748b", width=2, dash="dot"),
            ))
        fig.add_trace(go.Scatter(
            x=scen_df["Year"], y=scen_df["Scenario_Unemployment"],
            mode="lines+markers", name="AI Scenario",
            line=dict(color="#6366f1", width=3.5),
            marker=dict(size=7, color="#818cf8"),
            fill="tonexty" if not base_df.empty else "none",
            fillcolor="rgba(99,102,241,0.07)",
        ))

        # Color coded phases from story
        for event in story:
            yr  = event.get("year")
            phase = event.get("phase", "")
            if yr and "Shock" in phase:
                fig.add_vrect(x0=yr-0.4, x1=yr+0.4,
                              fillcolor="rgba(239,68,68,0.05)", line_width=0)

        fig.update_layout(**plotly_dark_layout(height=420))
        fig.update_layout(xaxis_title="Year", yaxis_title="Unemployment Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
