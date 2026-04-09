"""
Page 5 — AI Insights (Feature 4)
AI narrative panel, story timeline, macro/sector/recovery cards,
policy comparison, shock sensitivity analysis, and export report.
"""
import io
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui_helpers import (
    API_BASE_URL,
    DARK_CSS,
    plotly_dark_layout,
    render_badge,
    render_kpi_card,
)

st.set_page_config(page_title="AI Insights | UIP", page_icon="🤖", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 AI Insights")
    shock_intensity = st.slider("Shock Intensity", 0.0, 0.6, 0.3, 0.05)
    shock_duration  = st.slider("Shock Duration (yrs)", 0, 5, 2)
    recovery_rate   = st.slider("Recovery Rate", 0.05, 0.6, 0.3, 0.05)
    policy          = st.selectbox("Policy", ["None", "Fiscal Stimulus", "Monetary Policy", "Labor Reforms", "Industry Support"])
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
    st.page_link("pages/10_Skill_Obsolescence.py", label="⚡ Skill Obsolescence")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">🤖 AI Intelligence Engine</div>
    <div class="hero-subtitle">Machine-crafted economic narratives, year-by-year story mode, policy comparison, and sensitivity analysis</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.2);
            border-radius:14px; padding:1rem 1.4rem; margin-bottom:1.5rem;
            display:flex; align-items:flex-start; gap:1rem;">
    <div style="font-size:1.5rem; margin-top:0.1rem;">🧪</div>
    <div>
        <div style="font-size:0.78rem; font-weight:700; color:#818cf8; text-transform:uppercase;
                    letter-spacing:1px; margin-bottom:0.35rem;">Simulation Mode — AI narratives on projected data</div>
        <div style="font-size:0.87rem; color:#94a3b8; line-height:1.6;">
            The AI narrative engine generates insights from a
            <strong style="color:#e2e8f0;">simulated economic shock scenario</strong> (configured in the sidebar),
            not from live observed data. The LLM interprets model-generated unemployment trajectories
            and sector stress scores. Configure the scenario parameters and click
            <strong style="color:#e2e8f0;">Generate AI Insights</strong> to run analysis.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_insights_data(si, sd, rr, pol):
    try:
        r = requests.post(
            f"{API_BASE_URL}/simulate",
            json={"shock_intensity": si, "shock_duration": sd,
                  "recovery_rate": rr, "forecast_horizon": 7,
                  "policy_name": pol if pol != "None" else None},
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=90)
def get_policy_comparison_data(si, sd, rr):
    """Fetch two policy scenarios for side-by-side comparison."""
    policies = ["Fiscal Stimulus", "Labor Reforms"]
    results = {}
    for p in policies:
        try:
            r = requests.post(
                f"{API_BASE_URL}/simulate",
                json={"shock_intensity": si, "shock_duration": sd,
                      "recovery_rate": rr, "forecast_horizon": 7,
                      "policy_name": p},
                timeout=20,
            )
            if r.status_code == 200:
                results[p] = r.json()
        except Exception:
            pass
    return results


@st.cache_data(ttl=120)
def get_sensitivity_data(sd, rr):
    """Run simulations across shock intensities to build sensitivity table."""
    levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for si in levels:
        try:
            r = requests.post(
                f"{API_BASE_URL}/simulate",
                json={"shock_intensity": si, "shock_duration": sd,
                      "recovery_rate": rr, "forecast_horizon": 6},
                timeout=15,
            )
            if r.status_code == 200:
                d = r.json()
                idx = d.get("indices", {})
                rows.append({
                    "Shock Intensity": f"{int(si*100)}%",
                    "Peak Δ (pp)": idx.get("peak_delta", "—"),
                    "Stress Index": idx.get("unemployment_stress_index", "—"),
                    "Recovery Quality": idx.get("rqi_label", "—"),
                    "Early Warning": idx.get("early_warning", "—"),
                })
        except Exception:
            pass
    return rows


data = get_insights_data(shock_intensity, shock_duration, recovery_rate, policy)
if not data:
    st.error("⚠️ Cannot connect to API. Make sure the FastAPI Backend workflow is running.")
    st.stop()

insights = data.get("ai_insights", {})
story    = data.get("story", [])
indices  = data.get("indices", {})
scen_df  = pd.DataFrame(data.get("scenario", []))
source   = insights.get("source", "")

# ─── AI Provider Status ────────────────────────────────────────────────────────
import os as _os
_groq_set   = bool(_os.environ.get("GROQ_API_KEY", "").strip())
_gemini_set = bool(_os.environ.get("GEMINI_API_KEY", "").strip())
_openai_set = bool(_os.environ.get("OPENAI_API_KEY", "").strip())
_any_ai_set = _groq_set or _gemini_set or _openai_set

if not _any_ai_set:
    st.info(
        "**🔑 No AI key detected — using rule-based insights.**  \n"
        "To unlock real AI narratives for free, add one of these secrets:  \n"
        "• **GROQ_API_KEY** — free, no credit card · [console.groq.com](https://console.groq.com)  \n"
        "• **GEMINI_API_KEY** — free with Google account · [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)  \n"
        "• **OPENAI_API_KEY** — paid premium · [platform.openai.com](https://platform.openai.com)",
        icon="💡",
    )


# ─── KPI Row ──────────────────────────────────────────────────────────────────
rqi  = indices.get("rqi_label", "N/A")
usi  = indices.get("unemployment_stress_index", "N/A")
ew   = indices.get("early_warning", "N/A")
peak = round(scen_df["Scenario_Unemployment"].max(), 2) if not scen_df.empty else "N/A"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_kpi_card("🧠", "AI Risk Signal", ew.split(" ", 1)[-1] if " " in str(ew) else str(ew), delta_type="neutral"), unsafe_allow_html=True)
with c2:
    st.markdown(render_kpi_card("📊", "Stress Index", str(usi), delta_type="neutral"), unsafe_allow_html=True)
with c3:
    st.markdown(render_kpi_card("🔄", "Recovery Quality", str(rqi), delta_type="neutral"), unsafe_allow_html=True)
with c4:
    st.markdown(render_kpi_card("🎯", "Scenario Peak", f"{peak}%", delta_type="neutral"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── AI Summary Banner ────────────────────────────────────────────────────────
summary = insights.get("summary", "")
if "GPT" in source or "OpenAI" in source:
    source_badge_color = "blue"
elif "Groq" in source or "LLaMA" in source:
    source_badge_color = "green"
elif "Gemini" in source:
    source_badge_color = "purple"
else:
    source_badge_color = "yellow"
source_label = source if source else "📐 Rule-based"

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
            <div style="flex:1;">
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                    <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                                letter-spacing:1.5px; color:#818cf8;">AI ECONOMIC BRIEF</div>
                    <span class="badge badge-{source_badge_color}" style="font-size:0.68rem;">{source_label}</span>
                </div>
                <p style="color:#cbd5e1; font-size:1rem; line-height:1.7; margin:0;">{summary}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Insight Trio ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
insight_cards = [
    ("🌍", "Macro View",       "macro_insight",    "#818cf8"),
    ("🏭", "Sector View",      "sector_insight",   "#06b6d4"),
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
        _TYPE_ICON  = {"shock": "🔴", "recovery": "🔄", "stable": "🟢"}
        _TYPE_COLOR = {"shock": "#ef4444", "recovery": "#f59e0b", "stable": "#10b981"}
        for event in story:
            yr          = event.get("year", "?")
            val         = event.get("scenario_val", "?")
            desc        = event.get("body", "")
            event_type  = event.get("type", "stable")
            icon        = _TYPE_ICON.get(event_type, "📅")
            color       = _TYPE_COLOR.get(event_type, "#94a3b8")
            phase_label = event_type.capitalize()
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
                        <span class="badge badge-info" style="font-size:0.7rem; padding:0.1rem 0.5rem;">{phase_label}</span>
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

        for event in story:
            yr    = event.get("year")
            phase = event.get("type", "")
            if yr and phase == "shock":
                fig.add_vrect(x0=yr - 0.4, x1=yr + 0.4,
                              fillcolor="rgba(239,68,68,0.05)", line_width=0)

        fig.update_layout(**plotly_dark_layout(height=420))
        fig.update_layout(xaxis_title="Year", yaxis_title="Unemployment Rate (%)")
        st.plotly_chart(fig, width='stretch')

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Policy Comparison ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">⚖️ POLICY COMPARISON</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#64748b; font-size:0.88rem; margin-bottom:1.2rem;">
How do Fiscal Stimulus vs Labor Reforms change the AI narrative under your current shock parameters?
</p>
""", unsafe_allow_html=True)

with st.spinner("Running policy comparison..."):
    policy_data = get_policy_comparison_data(shock_intensity, shock_duration, recovery_rate)

if policy_data:
    pc1, pc2 = st.columns(2)
    palette = {"Fiscal Stimulus": "#6366f1", "Labor Reforms": "#10b981"}
    for col, (pol_name, pol_result) in zip([pc1, pc2], policy_data.items()):
        pol_insights = pol_result.get("ai_insights", {})
        pol_indices  = pol_result.get("indices", {})
        pol_ew       = pol_indices.get("early_warning", "N/A")
        pol_rqi      = pol_indices.get("rqi_label", "N/A")
        pol_usi      = pol_indices.get("unemployment_stress_index", "N/A")
        pol_summary  = pol_insights.get("summary", "No data.")
        color        = palette.get(pol_name, "#818cf8")
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                    <div style="width:10px; height:10px; border-radius:50%; background:{color};
                                box-shadow:0 0 6px {color};"></div>
                    <div style="font-size:1rem; font-weight:700; color:#e2e8f0;">{pol_name}</div>
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.8rem;">
                    <span class="badge badge-blue">USI: {pol_usi}</span>
                    <span class="badge badge-green">RQI: {pol_rqi}</span>
                    <span class="badge badge-purple">⚡ {pol_ew.split(" ",1)[-1] if " " in str(pol_ew) else pol_ew}</span>
                </div>
                <p style="color:#94a3b8; font-size:0.85rem; line-height:1.6; margin:0;">{pol_summary[:320]}{"..." if len(pol_summary) > 320 else ""}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Policy comparison data unavailable.")


# ─── Shock Sensitivity Analysis ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 SHOCK SENSITIVITY ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#64748b; font-size:0.88rem; margin-bottom:1.2rem;">
How does the AI risk signal and stress index change across shock intensities with your current duration and recovery settings?
</p>
""", unsafe_allow_html=True)

with st.spinner("Computing sensitivity table..."):
    sens_rows = get_sensitivity_data(shock_duration, recovery_rate)

if sens_rows:
    sens_df = pd.DataFrame(sens_rows)

    def _color_warning(val):
        if "High Risk" in str(val):
            return "background-color: rgba(239,68,68,0.15); color: #f87171;"
        elif "Watch" in str(val):
            return "background-color: rgba(245,158,11,0.12); color: #fbbf24;"
        elif "Stable" in str(val):
            return "background-color: rgba(16,185,129,0.10); color: #34d399;"
        return ""

    st.dataframe(
        sens_df.style.map(_color_warning, subset=["Early Warning"]),
        width='stretch',
        hide_index=True,
    )
else:
    st.info("Sensitivity data unavailable.")


# ─── Export Report ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📥 EXPORT INSIGHTS REPORT</div>', unsafe_allow_html=True)

def _build_report() -> bytes:
    lines = [
        "UNEMPLOYMENT INTELLIGENCE PLATFORM — AI INSIGHTS REPORT",
        "=" * 60,
        f"Scenario Parameters:",
        f"  Shock Intensity : {shock_intensity}",
        f"  Shock Duration  : {shock_duration} years",
        f"  Recovery Rate   : {recovery_rate}",
        f"  Policy          : {policy}",
        "",
        "KEY INDICES",
        "-" * 40,
        f"  Unemployment Stress Index : {usi}",
        f"  Recovery Quality          : {rqi}",
        f"  Early Warning Status      : {ew}",
        f"  Scenario Peak Rate        : {peak}%",
        "",
        "AI ECONOMIC BRIEF",
        "-" * 40,
        summary,
        "",
        "MACRO VIEW",
        "-" * 40,
        insights.get("macro_insight", ""),
        "",
        "SECTOR VIEW",
        "-" * 40,
        insights.get("sector_insight", ""),
        "",
        "RECOVERY OUTLOOK",
        "-" * 40,
        insights.get("recovery_insight", ""),
        "",
        "YEAR-BY-YEAR STORY",
        "-" * 40,
    ]
    for event in story:
        lines.append(f"  {event.get('year','?')} | {event.get('type','').upper():8s} | {event.get('scenario_val','?')}% | {event.get('body','')}")
    if sens_rows:
        lines += ["", "SHOCK SENSITIVITY ANALYSIS", "-" * 40]
        for row in sens_rows:
            lines.append(f"  {row['Shock Intensity']:>6} | Stress: {row['Stress Index']:>5} | RQI: {row['Recovery Quality']:<22} | {row['Early Warning']}")
    lines += ["", f"Report source: {source_label}", "Generated by Unemployment Intelligence Platform"]
    return "\n".join(lines).encode("utf-8")

report_bytes = _build_report()
st.download_button(
    label="⬇️ Download Full Insights Report (.txt)",
    data=report_bytes,
    file_name=f"uip_insights_shock{int(shock_intensity*100)}pct.txt",
    mime="text/plain",
    use_container_width=False,
)
