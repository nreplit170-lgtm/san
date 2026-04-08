"""
Page 2 — Scenario Simulator
Side-by-side scenario lab with live comparison charts, gauge, and metrics.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout, API_BASE_URL

st.set_page_config(page_title="Simulator | UIP", page_icon="🧪", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─── Sidebar nav ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧪 Simulator")
    st.markdown("Configure two scenarios and compare their outcomes.")
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")
    st.page_link("pages/10_Skill_Obsolescence.py", label="⚡ Skill Obsolescence")

# ─── Page hero ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="hero-title">🧪 Scenario Simulator Lab</div>
    <div class="hero-subtitle">Design economic shock scenarios and compare their unemployment trajectories side-by-side</div>
</div>""", unsafe_allow_html=True)

SCENARIO_PRESETS = {
    "Baseline (Natural Flow)":  dict(si=0.0, sd=0, rr=0.0),
    "Severe Economic Shock":    dict(si=0.5, sd=3, rr=0.2),
    "Moderate Recession":       dict(si=0.3, sd=2, rr=0.3),
    "Policy Intervention":      dict(si=0.2, sd=2, rr=0.45),
    "Global Crisis":             dict(si=0.6, sd=4, rr=0.15),
    "Rapid Recovery":           dict(si=0.2, sd=1, rr=0.55),
}
POLICY_OPTIONS = ["None", "Fiscal Stimulus", "Monetary Policy", "Labor Reforms", "Industry Support"]

# ─── Configuration Panels ──────────────────────────────────────────────────────
col_cfg_a, col_vs, col_cfg_b = st.columns([5, 1, 5])

with col_cfg_a:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Scenario A — Configure</div>', unsafe_allow_html=True)
    preset_a = st.selectbox("Quick Preset", list(SCENARIO_PRESETS.keys()), key="preset_a")
    pa = SCENARIO_PRESETS[preset_a]
    si_a = st.slider("Shock Intensity", 0.0, 0.6, float(pa["si"]), 0.05, key="si_a",
                     help="How severe the economic shock is (0 = none, 0.6 = catastrophic)")
    sd_a = st.slider("Shock Duration (yrs)", 0, 5, int(pa["sd"]), key="sd_a")
    rr_a = st.slider("Recovery Rate", 0.05, 0.6, max(0.05, float(pa["rr"])), 0.05, key="rr_a")
    policy_a = st.selectbox("Policy Response", POLICY_OPTIONS, key="pol_a")
    horizon_a = st.slider("Forecast Horizon (yrs)", 3, 10, 6, key="horiz_a")
    st.markdown("</div>", unsafe_allow_html=True)

with col_vs:
    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; background:linear-gradient(135deg,#6366f1,#8b5cf6);
                width:44px; height:44px; border-radius:50%; display:flex; align-items:center;
                justify-content:center; margin:0 auto; font-size:1rem; font-weight:800;
                color:white; box-shadow:0 8px 24px rgba(99,102,241,0.4);">VS</div>
    """, unsafe_allow_html=True)

with col_cfg_b:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Scenario B — Configure</div>', unsafe_allow_html=True)
    preset_b = st.selectbox("Quick Preset", list(SCENARIO_PRESETS.keys()), index=1, key="preset_b")
    pb = SCENARIO_PRESETS[preset_b]
    si_b = st.slider("Shock Intensity", 0.0, 0.6, float(pb["si"]), 0.05, key="si_b")
    sd_b = st.slider("Shock Duration (yrs)", 0, 5, int(pb["sd"]), key="sd_b")
    rr_b = st.slider("Recovery Rate", 0.05, 0.6, max(0.05, float(pb["rr"])), 0.05, key="rr_b")
    policy_b = st.selectbox("Policy Response", POLICY_OPTIONS, index=1, key="pol_b")
    horizon_b = st.slider("Forecast Horizon (yrs)", 3, 10, 6, key="horiz_b")
    st.markdown("</div>", unsafe_allow_html=True)

horizon = max(horizon_a, horizon_b)

# ─── Run Button ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_run = st.columns([3, 2, 3])
with col_run[1]:
    run = st.button("🚀 Run Simulation", use_container_width=True)

# ─── API helper ───────────────────────────────────────────────────────────────
def fetch(si, sd, rr, pol, h):
    try:
        r = requests.post(f"{API_BASE_URL}/simulate",
                          json={"shock_intensity": si, "shock_duration": sd,
                                "recovery_rate": rr, "forecast_horizon": h,
                                "policy_name": pol if pol != "None" else None},
                          timeout=25)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None

# ─── Simulation state ─────────────────────────────────────────────────────────
if run or "sim_a" not in st.session_state:
    with st.spinner("⚡ Running simulations..."):
        st.session_state.sim_baseline = fetch(0.0, 0, 0.0, "None", horizon)
        st.session_state.sim_a = fetch(si_a, sd_a, rr_a, policy_a, horizon)
        st.session_state.sim_b = fetch(si_b, sd_b, rr_b, policy_b, horizon)

sim_base = st.session_state.get("sim_baseline")
sim_a    = st.session_state.get("sim_a")
sim_b    = st.session_state.get("sim_b")

if not (sim_base and sim_a and sim_b):
    st.warning("Run the simulation above or start the API backend.")
    st.stop()

base_df  = pd.DataFrame(sim_base["baseline"])
scen_a_df = pd.DataFrame(sim_a["scenario"])
scen_b_df = pd.DataFrame(sim_b["scenario"])
idx_a = sim_a.get("indices", {})
idx_b = sim_b.get("indices", {})

peak_base = round(base_df["Predicted_Unemployment"].max(), 2)
peak_a = round(scen_a_df["Scenario_Unemployment"].max(), 2)
peak_b = round(scen_b_df["Scenario_Unemployment"].max(), 2)
delta_a = round(peak_a - peak_base, 2)
delta_b = round(peak_b - peak_base, 2)

# ─── Result KPIs ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(render_kpi_card("📊", "Baseline Peak", f"{peak_base}%", delta_type="neutral"), unsafe_allow_html=True)
with c2:
    dt = "up" if delta_a > 0 else "down"
    st.markdown(render_kpi_card("🔵", "Scenario A Peak", f"{peak_a}%", f"{'▲' if delta_a>0 else '▼'} {abs(delta_a)}pp", dt), unsafe_allow_html=True)
with c3:
    dt = "up" if delta_b > 0 else "down"
    st.markdown(render_kpi_card("🟣", "Scenario B Peak", f"{peak_b}%", f"{'▲' if delta_b>0 else '▼'} {abs(delta_b)}pp", dt), unsafe_allow_html=True)
with c4:
    ew_a = idx_a.get("early_warning", "N/A")
    st.markdown(render_kpi_card("🚦", "A — Risk", ew_a.split(" ",1)[-1] if " " in ew_a else ew_a, delta_type="neutral"), unsafe_allow_html=True)
with c5:
    ew_b = idx_b.get("early_warning", "N/A")
    st.markdown(render_kpi_card("🚦", "B — Risk", ew_b.split(" ",1)[-1] if " " in ew_b else ew_b, delta_type="neutral"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Comparison Chart ─────────────────────────────────────────────────────────
col_main, col_side = st.columns([3, 1])

with col_main:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Trajectory Comparison</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base_df["Year"], y=base_df["Predicted_Unemployment"],
        mode="lines", name="Baseline",
        line=dict(color="#64748b", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=scen_a_df["Year"], y=scen_a_df["Scenario_Unemployment"],
        mode="lines+markers", name=f"Scenario A · {policy_a}",
        line=dict(color="#6366f1", width=3.5),
        marker=dict(size=7, color="#818cf8"),
        fill="tonexty", fillcolor="rgba(99,102,241,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=scen_b_df["Year"], y=scen_b_df["Scenario_Unemployment"],
        mode="lines+markers", name=f"Scenario B · {policy_b}",
        line=dict(color="#8b5cf6", width=3.5),
        marker=dict(size=7, color="#a78bfa"),
    ))
    fig.update_layout(**plotly_dark_layout(height=380))
    fig.update_layout(xaxis_title="Year", yaxis_title="Unemployment Rate (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_side:
    st.markdown('<div class="glass-card" style="height:100%">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Peak Bar</div>', unsafe_allow_html=True)
    bar_df = pd.DataFrame({
        "Scenario": ["Baseline", "Scenario A", "Scenario B"],
        "Peak %": [peak_base, peak_a, peak_b],
        "Color": ["#64748b", "#6366f1", "#8b5cf6"]
    })
    fig_b = go.Figure(go.Bar(
        x=bar_df["Peak %"], y=bar_df["Scenario"],
        orientation="h",
        marker_color=bar_df["Color"].tolist(),
        text=[f"{v:.2f}%" for v in bar_df["Peak %"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig_b.update_layout(**plotly_dark_layout(height=220))
    fig_b.update_layout(xaxis_title="Peak %", margin=dict(l=5, r=40, t=5, b=5), showlegend=False)
    st.plotly_chart(fig_b, use_container_width=True)

    # Risk gauges
    st.markdown("<br>", unsafe_allow_html=True)
    for label, peak_val in [("Scenario A", peak_a), ("Scenario B", peak_b)]:
        color = "#ef4444" if peak_val > 7 else "#f59e0b" if peak_val > 5 else "#10b981"
        pct = min(int((peak_val / 12) * 100), 100)
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                <span style="color:#94a3b8; font-size:0.8rem; font-weight:600;">{label}</span>
                <span style="color:{color}; font-size:0.8rem; font-weight:700;">{peak_val}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:999px; height:8px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:{color}; border-radius:999px;
                            box-shadow:0 0 8px {color}; transition:width 0.6s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Parameter + Indices table ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_p, col_i = st.columns(2)

with col_p:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Parameter Comparison</div>', unsafe_allow_html=True)
    pdata = {
        "Parameter": ["Shock Intensity", "Shock Duration", "Recovery Rate", "Policy"],
        "Scenario A": [f"{si_a*100:.0f}%", f"{sd_a} yrs", f"{rr_a*100:.0f}%", policy_a],
        "Scenario B": [f"{si_b*100:.0f}%", f"{sd_b} yrs", f"{rr_b*100:.0f}%", policy_b],
    }
    st.dataframe(pd.DataFrame(pdata), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_i:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Scenario Indices</div>', unsafe_allow_html=True)
    idata = {
        "Index": ["Stress Index (USI)", "Recovery Quality", "Policy Cushion", "Early Warning"],
        "Scenario A": [
            idx_a.get("unemployment_stress_index","N/A"),
            idx_a.get("rqi_label","N/A"),
            idx_a.get("policy_cushion_score","N/A"),
            idx_a.get("early_warning","N/A"),
        ],
        "Scenario B": [
            idx_b.get("unemployment_stress_index","N/A"),
            idx_b.get("rqi_label","N/A"),
            idx_b.get("policy_cushion_score","N/A"),
            idx_b.get("early_warning","N/A"),
        ],
    }
    st.dataframe(pd.DataFrame(idata), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
