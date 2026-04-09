"""
Home Page — Unemployment Intelligence Platform
Premium dark glassmorphism landing experience.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout, API_BASE_URL

st.set_page_config(
    page_title="Unemployment Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─── Extra landing-page CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-wrap {
    text-align: center;
    padding: 4rem 2rem 3rem;
}
.hero-super {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #818cf8 !important;
    margin-bottom: 1.5rem;
}
.hero-main-title {
    font-size: 4rem !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
    background: linear-gradient(135deg, #f1f5f9 0%, #818cf8 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.2rem !important;
}
.hero-desc {
    font-size: 1.15rem !important;
    color: #94a3b8 !important;
    max-width: 600px;
    margin: 0 auto 2.5rem !important;
    line-height: 1.7;
}
.nav-link-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    transition: all 0.25s ease;
    cursor: pointer;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.7rem;
    position: relative;
    overflow: hidden;
}
.nav-link-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
.nav-link-card:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
    transform: translateY(-5px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(99,102,241,0.2);
}
.nav-icon { font-size: 2.4rem; }
.nav-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0 !important;
}
.nav-desc {
    font-size: 0.82rem;
    color: #64748b !important;
    line-height: 1.4;
}
.status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    padding: 0.9rem 2rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 999px;
    margin: 0 auto 2.5rem;
    max-width: 600px;
}
.status-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: #64748b !important;
    font-weight: 500;
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-green { background: #10b981; box-shadow: 0 0 6px #10b981; }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
.dot-yellow { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─── API health check ──────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def check_api_health():
    last_err = "Unknown error"
    for attempt in range(2):
        try:
            r = requests.get(f"{API_BASE_URL}/data-status", timeout=5)
            if r.status_code == 200:
                payload = r.json()
                return True, None, payload.get("source", "")
        except Exception as e:
            last_err = str(e)
    return False, last_err, ""

@st.cache_data(ttl=60)
def get_baseline_preview():
    try:
        payload = {"shock_intensity": 0.0, "shock_duration": 0, "recovery_rate": 0.0, "forecast_horizon": 6}
        r = requests.post(f"{API_BASE_URL}/simulate", json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

api_ok, api_err, data_source_label = check_api_health()
baseline_preview = get_baseline_preview()

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-super">🌐 Economic Intelligence Platform v2.0</div>
    <div class="hero-main-title">Unemployment<br>Intelligence Platform</div>
    <p class="hero-desc">
        Scenario-based forecasting, shock simulation, and policy analysis — 
        built for economists, policymakers, and data scientists.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Status Bar ───────────────────────────────────────────────────────────────
api_dot   = "dot-green" if api_ok else "dot-red"
api_label = "API Online" if api_ok else "API Offline"

# Derive World Bank status from the live /data-status response
_src_lower = data_source_label.lower()
if "live" in _src_lower or "world bank" in _src_lower:
    wb_dot, wb_label = "dot-green", "World Bank Live"
elif "offline" in _src_lower or "csv" in _src_lower:
    wb_dot, wb_label = "dot-yellow", "Offline (CSV Fallback)"
elif api_ok:
    wb_dot, wb_label = "dot-yellow", "Data Loading…"
else:
    wb_dot, wb_label = "dot-red", "Data Unavailable"

st.markdown(f"""
<div class="status-bar">
    <div class="status-item"><span class="dot {api_dot}"></span>{api_label}</div>
    <div class="status-item"><span class="dot dot-green"></span>India Dataset Loaded</div>
    <div class="status-item"><span class="dot {wb_dot}"></span>{wb_label}</div>
    <div class="status-item"><span class="dot dot-yellow"></span>Live Forecasting</div>
</div>
""", unsafe_allow_html=True)

if not api_ok:
    st.warning(f"⚠️ FastAPI backend is not running. Start it with: `uvicorn src.api:app --reload`  \nError: {api_err}")

# ─── KPI Row ──────────────────────────────────────────────────────────────────
if baseline_preview:
    baseline_df = pd.DataFrame(baseline_preview.get("baseline", []))
    if not baseline_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        peak = round(baseline_df["Predicted_Unemployment"].max(), 2)
        peak_year = int(baseline_df.loc[baseline_df["Predicted_Unemployment"].idxmax(), "Year"])
        current = round(baseline_df["Predicted_Unemployment"].iloc[0], 2)
        end_val = round(baseline_df["Predicted_Unemployment"].iloc[-1], 2)

        with col1:
            st.markdown(render_kpi_card("📊", "Current Rate", f"{current}%", delta_type="neutral"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_kpi_card("🎯", "Baseline Peak", f"{peak}%", f"in {peak_year}", "up"), unsafe_allow_html=True)
        with col3:
            delta_6y = round(end_val - current, 2)
            dtype = "up" if delta_6y > 0 else "down"
            st.markdown(render_kpi_card("📉", "6-Year Outlook", f"{end_val}%", f"{'▲' if delta_6y>0 else '▼'} {abs(delta_6y)}pp", dtype), unsafe_allow_html=True)
        with col4:
            indices = baseline_preview.get("indices", {})
            ew = indices.get("early_warning", "🟢 Stable")
            st.markdown(render_kpi_card("🚦", "Risk Status", ew.split(" ", 1)[-1] if " " in ew else ew, delta_type="neutral"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mini sparkline chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=baseline_df["Year"],
            y=baseline_df["Predicted_Unemployment"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            line=dict(color="#6366f1", width=3),
            name="Baseline Forecast",
        ))
        fig.update_layout(**plotly_dark_layout(height=200))
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        fig.update_xaxes(showgrid=False, showticklabels=True, linecolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", title_text="Unemployment %")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<div style='color:#94a3b8; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>📈 BASELINE FORECAST PREVIEW</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

# ─── Navigation Cards ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; color:#6366f1; margin-bottom:1.5rem;'>↓ EXPLORE THE PLATFORM</div>", unsafe_allow_html=True)

pages = [
    ("📊", "Overview Dashboard", "Live KPIs, forecast trajectories & historical event overlays", "pages/1_Overview.py"),
    ("🧪", "Scenario Simulator", "Design & compare two economic shock scenarios", "pages/2_Simulator.py"),
    ("🏭", "Sector Analysis", "Heatmap, radar chart & sector resilience breakdown", "pages/3_Sector_Analysis.py"),
    ("💼", "Career Lab", "Skill demand, growth sectors & career path guidance", "pages/4_Career_Lab.py"),
    ("🤖", "AI Insights", "AI-generated narratives & story-mode timeline", "pages/5_AI_Insights.py"),
    ("🔬", "Model Validation", "Backtest accuracy, R² score & reliability metrics", "pages/6_Model_Validation.py"),
    ("🎯", "Job Risk (AI)", "ML risk score from skills, education, experience & location", "pages/7_Job_Risk_Predictor.py"),
    ("📡", "Job Market Pulse", "Skill & role demand trends from job postings (CSV)", "pages/8_Job_Market_Pulse.py"),
    ("🗺️", "Geo Career", "Maps, location quotients & relocation signals from your data", "pages/9_Geo_Career_Advisor.py"),
    ("📉", "Skill Obsolescence", "Declining vs emerging skills from posting history", "pages/10_Skill_Obsolescence.py"),
]

col_grid = st.columns(3)
for i, (icon, title, desc, path) in enumerate(pages):
    with col_grid[i % 3]:
        st.markdown(f"""
        <div class="nav-link-card">
            <div class="nav-icon">{icon}</div>
            <div class="nav-title">{title}</div>
            <div class="nav-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        # Use a hidden/small page link button below the card
        st.page_link(path, label=f"Go to {title}", use_container_width=True)
        if i % 3 != 2:
            st.markdown("")

# ─── Data Modes Explanation ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
            border-radius:20px; padding:2rem 2.5rem; margin-bottom:2rem;">
    <div style="text-align:center; font-size:0.78rem; font-weight:700; text-transform:uppercase;
                letter-spacing:2px; color:#6366f1; margin-bottom:1.5rem;">
        ↕ TWO DATA MODES IN ONE PLATFORM
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
        <div style="background:rgba(6,182,212,0.06); border:1px solid rgba(6,182,212,0.2);
                    border-radius:14px; padding:1.4rem;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.7rem;">
                <span style="font-size:1.2rem;">🌐</span>
                <span style="font-size:0.85rem; font-weight:700; color:#06b6d4; text-transform:uppercase;
                              letter-spacing:0.8px;">Real Data Mode</span>
            </div>
            <div style="font-size:0.87rem; color:#94a3b8; line-height:1.7;">
                Pulls <strong style="color:#e2e8f0;">live unemployment, employment, and GDP indicators</strong>
                directly from the World Bank Open Data API (no key required). Figures are the most recently
                published annual values. Used on:
                <ul style="margin:0.5rem 0 0; padding-left:1.2rem; color:#cbd5e1;">
                    <li>Overview → Evidence-Based Forecast</li>
                    <li>Sector Analysis → Live World Bank Data</li>
                    <li>Market Pulse → Live India Labor Data</li>
                    <li>Geo Career → Live indicators</li>
                </ul>
            </div>
        </div>
        <div style="background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.2);
                    border-radius:14px; padding:1.4rem;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.7rem;">
                <span style="font-size:1.2rem;">🧪</span>
                <span style="font-size:0.85rem; font-weight:700; color:#818cf8; text-transform:uppercase;
                              letter-spacing:0.8px;">Simulation Mode</span>
            </div>
            <div style="font-size:0.87rem; color:#94a3b8; line-height:1.7;">
                Uses <strong style="color:#e2e8f0;">parametric shock equations seeded from India's historical
                baseline</strong> (~7% structural unemployment rate) to generate hypothetical scenarios.
                You control shock intensity, duration, recovery rate, and policy response. Used on:
                <ul style="margin:0.5rem 0 0; padding-left:1.2rem; color:#cbd5e1;">
                    <li>Scenario Simulator</li>
                    <li>Career Lab</li>
                    <li>AI Insights (LLM on simulation output)</li>
                    <li>Sector Analysis → Scenario tab</li>
                </ul>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.8rem; padding:1rem 0; border-top:1px solid rgba(255,255,255,0.05);">
    Built by <strong style="color:#6366f1;">Bhushan Nanavare</strong> · 
    Unemployment Intelligence Platform · 
    Data: World Bank Open Data
</div>
""", unsafe_allow_html=True)
