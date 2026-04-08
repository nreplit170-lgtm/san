"""
Page 3 — Sector Analysis
Heatmap, radar chart, animated treemap, and sector resilience cards.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout, API_BASE_URL

st.set_page_config(page_title="Sector Analysis | UIP", page_icon="🏭", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏭 Sector Analysis")
    shock_intensity = st.slider("Shock Intensity", 0.0, 0.6, 0.3, 0.05)
    recovery_rate   = st.slider("Recovery Rate", 0.05, 0.6, 0.3, 0.05)
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/2_Simulator.py", label="🧪 Simulator")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">🏭 Sector Intelligence</div>
    <div class="hero-subtitle">Stress levels, resilience scores, and cross-sector impact analysis</div>
</div>""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_sector_data(si, rr):
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

if st.sidebar.button("🔄 Refresh Analysis"):
    st.cache_data.clear()

data = get_sector_data(shock_intensity, recovery_rate)

if not data:
    st.error("⚠️ Cannot connect to API. Start: `uvicorn src.api:app --reload`")
    st.stop()

sector_raw = data.get("sector_impact", [])
if not sector_raw:
    st.warning("No sector data returned. Try adjusting parameters.")
    st.stop()

df = pd.DataFrame(sector_raw)

# ─── Summary KPIs ──────────────────────────────────────────────────────────────
most_stressed = df.loc[df["Stress_Score"].idxmax(), "Sector"]
most_resilient = df.loc[df["Resilience_Score"].idxmax(), "Sector"]
avg_stress = round(df["Stress_Score"].mean(), 2)
high_risk_count = len(df[df["Stress_Score"] >= 60.0])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_kpi_card("🔥", "Most Stressed", most_stressed, delta_type="neutral"), unsafe_allow_html=True)
with c2:
    st.markdown(render_kpi_card("💪", "Most Resilient", most_resilient, delta_type="neutral"), unsafe_allow_html=True)
with c3:
    at = "up" if avg_stress > 0.4 else "down"
    st.markdown(render_kpi_card("📊", "Avg Stress Score", f"{avg_stress:.2f}", delta_type=at), unsafe_allow_html=True)
with c4:
    bt = "up" if high_risk_count > 2 else "neutral"
    st.markdown(render_kpi_card("⚠️", "High-Risk Sectors", str(high_risk_count), delta_type=bt), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Charts Row ──────────────────────────────────────────────────────────
col_heat, col_radar = st.columns([3, 2])

with col_heat:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌡️ Stress vs Resilience Heatmap</div>', unsafe_allow_html=True)
    heat_df = df[["Sector", "Stress_Score", "Resilience_Score"]].copy()
    heat_matrix = heat_df.set_index("Sector").T

    fig_heat = go.Figure(go.Heatmap(
        z=heat_matrix.values.tolist(),
        x=heat_matrix.columns.tolist(),
        y=["Stress Score", "Resilience Score"],
        colorscale=[
            [0.0, "#0a0e1a"],
            [0.3, "#1e3a5f"],
            [0.6, "#6366f1"],
            [0.8, "#f59e0b"],
            [1.0, "#ef4444"],
        ],
        showscale=True,
        text=[[f"{v:.2f}" for v in row] for row in heat_matrix.values.tolist()],
        texttemplate="%{text}",
        textfont=dict(color="white", size=11),
        hovertemplate="<b>%{x}</b><br>%{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color="#94a3b8"),
            title=dict(text="Score", font=dict(color="#94a3b8")),
        ),
    ))
    fig_heat.update_layout(**plotly_dark_layout(height=280))
    fig_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_radar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕸️ Resilience Radar</div>', unsafe_allow_html=True)
    sectors = df["Sector"].tolist()
    resilience_vals = df["Resilience_Score"].tolist()
    stress_vals = df["Stress_Score"].tolist()

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=resilience_vals + [resilience_vals[0]],
        theta=sectors + [sectors[0]],
        fill="toself",
        name="Resilience",
        fillcolor="rgba(16,185,129,0.15)",
        line=dict(color="#10b981", width=2),
        marker=dict(color="#10b981", size=5),
    ))
    fig_r.add_trace(go.Scatterpolar(
        r=stress_vals + [stress_vals[0]],
        theta=sectors + [sectors[0]],
        fill="toself",
        name="Stress",
        fillcolor="rgba(239,68,68,0.1)",
        line=dict(color="#ef4444", width=2, dash="dot"),
        marker=dict(color="#ef4444", size=5),
    ))
    fig_r.update_layout(
        **plotly_dark_layout(height=280),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(color="#64748b", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(color="#94a3b8", size=10)),
        ),
        showlegend=True,
    )
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Treemap ──────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🗺️ Sector Impact Treemap</div>', unsafe_allow_html=True)
fig_tree = px.treemap(
    df, path=["Sector"], values="Stress_Score",
    color="Stress_Score",
    color_continuous_scale=["#0d1b2a", "#1e3a5f", "#6366f1", "#f59e0b", "#ef4444"],
    hover_data={"Resilience_Score": ":.3f", "Stress_Score": ":.3f"},
)
fig_tree.update_traces(
    textfont=dict(color="white", size=13),
    marker=dict(cornerradius=5),
)
fig_tree.update_layout(
    **plotly_dark_layout(height=320),
    coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8"), title=dict(text="Stress", font=dict(color="#94a3b8"))),
)
st.plotly_chart(fig_tree, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─── Sector Cards ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">🏷️ Sector Detail Cards</div>', unsafe_allow_html=True)

cols = st.columns(min(len(df), 4))
for i, row in df.iterrows():
    col = cols[i % len(cols)]
    stress = row["Stress_Score"]
    res    = row["Resilience_Score"]
    if stress >= 0.6:
        badge = render_badge("🔴 High Risk", "danger")
    elif stress >= 0.35:
        badge = render_badge("🟡 Moderate", "warning")
    else:
        badge = render_badge("🟢 Stable", "success")

    sp = int(stress * 100)
    rp = int(res * 100)

    with col:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.8rem;">
                <div style="font-size:1rem; font-weight:700; color:#e2e8f0;">{row["Sector"]}</div>
                {badge}
            </div>
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8; margin-bottom:3px;">
                    <span>Stress</span><span style="color:#ef4444;">{stress:.2f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:6px;">
                    <div style="width:{sp}%; height:100%; background:#ef4444; border-radius:4px;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8; margin-bottom:3px;">
                    <span>Resilience</span><span style="color:#10b981;">{res:.2f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:6px;">
                    <div style="width:{rp}%; height:100%; background:#10b981; border-radius:4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
