"""
Page 1 — Overview Dashboard
Live KPIs, forecast trajectory with confidence bands, historical event overlays.
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout, API_BASE_URL

st.set_page_config(page_title="Overview | UIP", page_icon="📊", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def get_baseline(horizon: int):
    try:
        r = requests.post(f"{API_BASE_URL}/simulate",
                          json={"shock_intensity": 0.0, "shock_duration": 0,
                                "recovery_rate": 0.0, "forecast_horizon": horizon},
                          timeout=20)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# ─── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Overview Controls")
    horizon = st.slider("Forecast Horizon (years)", 3, 10, 6)
    show_events = st.checkbox("Show Historical Events", value=True)
    show_band = st.checkbox("Show Uncertainty Band", value=True)
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/2_Simulator.py", label="🧪 Scenario Simulator")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")

# ─── Page hero ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="hero-title">📊 Live Overview Dashboard</div>
    <div class="hero-subtitle">Real-time baseline forecast with uncertainty bands and historical event markers</div>
</div>""", unsafe_allow_html=True)

data = get_baseline(horizon)

if not data:
    st.error("⚠️ Cannot connect to API. Start: `uvicorn src.api:app --reload`")
    st.stop()

baseline_df = pd.DataFrame(data["baseline"])
indices = data.get("indices", {})

# ─── KPI Row ────────────────────────────────────────────────────────────────────
peak = round(baseline_df["Predicted_Unemployment"].max(), 2)
peak_year = int(baseline_df.loc[baseline_df["Predicted_Unemployment"].idxmax(), "Year"])
current = round(baseline_df["Predicted_Unemployment"].iloc[0], 2)
end_val = round(baseline_df["Predicted_Unemployment"].iloc[-1], 2)
ew = indices.get("early_warning", "🟢 Stable")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(render_kpi_card("📈", "Baseline Start", f"{current}%", delta_type="neutral"), unsafe_allow_html=True)
with col2:
    d = round(peak - current, 2)
    st.markdown(render_kpi_card("🎯", "Forecast Peak", f"{peak}%", f"▲ {d}pp · {peak_year}", "up"), unsafe_allow_html=True)
with col3:
    d6 = round(end_val - current, 2)
    dtype = "up" if d6 > 0 else "down"
    st.markdown(render_kpi_card("📉", f"{horizon}-Year Outlook", f"{end_val}%", f"{'▲' if d6>0 else '▼'} {abs(d6)}pp", dtype), unsafe_allow_html=True)
with col4:
    label = ew.split(" ", 1)[-1] if " " in ew else ew
    badge_kind = "danger" if "High" in ew else "warning" if "Watch" in ew else "success"
    delta_html = render_badge(ew, badge_kind)
    st.markdown(render_kpi_card("🚦", "Risk Status", label, delta_type="neutral"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Chart ─────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Unemployment Forecast Trajectory</div>', unsafe_allow_html=True)

years = baseline_df["Year"].tolist()
values = baseline_df["Predicted_Unemployment"].tolist()

# Generate uncertainty bands (±1σ, ±2σ)
sigma = np.linspace(0.1, 0.6, len(years))

fig = go.Figure()

if show_band:
    # +2σ upper → fill to lower boundary (upper-2σ to lower-2σ outer band)
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=[v + 2*s for v, s in zip(values, sigma)] + [v - 2*s for v, s in zip(values[::-1], sigma[::-1])],
        fill="toself",
        fillcolor="rgba(99,102,241,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        hoverinfo="skip",
        showlegend=True,
    ))
    # +1σ inner band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=[v + s for v, s in zip(values, sigma)] + [v - s for v, s in zip(values[::-1], sigma[::-1])],
        fill="toself",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="68% CI",
        hoverinfo="skip",
        showlegend=True,
    ))

fig.add_trace(go.Scatter(
    x=years, y=values,
    mode="lines+markers",
    name="Baseline Forecast",
    line=dict(color="#6366f1", width=3.5),
    marker=dict(size=7, color="#818cf8", line=dict(color="#6366f1", width=2)),
    hovertemplate="<b>Year %{x}</b><br>Unemployment: %{y:.2f}%<extra></extra>",
))

# Historical events overlay
if show_events:
    events = [
        (2008, 8.0, "2008 GFC"),
        (2020, 7.5, "COVID-19"),
        (2016, 6.0, "Demonetization"),
    ]
    for year, y_pos, label in events:
        if min(years) <= year <= max(years):
            fig.add_vline(x=year, line=dict(color="rgba(245,158,11,0.4)", width=1.5, dash="dot"))
            fig.add_annotation(
                x=year, y=y_pos, text=label,
                showarrow=False, font=dict(size=10, color="#f59e0b"),
                bgcolor="rgba(245,158,11,0.12)", bordercolor="rgba(245,158,11,0.3)",
                borderwidth=1, borderpad=4,
            )

fig.update_layout(**plotly_dark_layout(height=420))
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Unemployment Rate (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0.3)", font=dict(color="#cbd5e1")),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─── Metrics + Table ──────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1])

with col_l:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Scenario Indices</div>', unsafe_allow_html=True)
    idx_rows = [
        ("Unemployment Stress Index", indices.get("unemployment_stress_index", "N/A")),
        ("Recovery Quality Index", indices.get("rqi_label", "N/A")),
        ("Policy Cushion Score", indices.get("policy_cushion_score", "N/A")),
        ("Peak Delta (pp)", indices.get("peak_delta", "N/A")),
        ("Early Warning", indices.get("early_warning", "N/A")),
    ]
    for label, val in idx_rows:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:#94a3b8; font-size:0.88rem;">{label}</span>
            <span style="color:#e2e8f0; font-weight:700; font-size:0.88rem;">{val}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Forecast Data Table</div>', unsafe_allow_html=True)
    display_df = baseline_df[["Year", "Predicted_Unemployment"]].rename(
        columns={"Predicted_Unemployment": "Unemployment Rate (%)"}
    ).round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)
