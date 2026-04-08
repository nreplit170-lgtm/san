"""
Page 7 — AI-Based Job Risk Predictor (Feature 1)

User profile → ML risk estimate, explanations, and what-if skill simulation.
Runs entirely in-process (no FastAPI required).
"""
import streamlit as st
import plotly.graph_objects as go

from src.job_risk_model import (
    EDUCATION_LEVELS,
    FEATURE_NAMES,
    INDUSTRY_GROWTH,
    LOCATION_OPTIONS,
    predict_job_risk,
    what_if_improve_skills,
)
from src.ui_helpers import DARK_CSS, plotly_dark_layout

st.set_page_config(page_title="Job Risk (AI) | UIP", page_icon="🎯", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎯 Job Risk Predictor")
    st.caption("Logistic regression on structured features. No backend needed.")
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/2_Simulator.py", label="🧪 Simulator")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/6_Model_Validation.py", label="🔬 Model Validation")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">🎯 AI Job Risk Predictor</div>
    <div class="hero-subtitle">
        Estimate unemployment-risk probability from skills, education, experience, industry growth, and location —
        with reasons, suggestions, and what-if skill upgrades.
    </div>
</div>""", unsafe_allow_html=True)

col_form, col_out = st.columns([1, 1])

with col_form:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Your profile</div>', unsafe_allow_html=True)
    skills = st.text_area(
        "Skills (comma-separated)",
        placeholder="e.g. Python, SQL, cloud computing, communication",
        height=100,
        help="We match phrases to an in-house demand lexicon to build a skill-demand score.",
    )
    education = st.selectbox("Education", EDUCATION_LEVELS, index=2)
    experience = st.slider("Years of experience", 0, 40, 3)
    industry = st.selectbox("Industry / sector", list(INDUSTRY_GROWTH.keys()))
    location = st.selectbox("Location (optional context)", LOCATION_OPTIONS)
    run = st.button("🔮 Estimate risk", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What-if: add skills</div>', unsafe_allow_html=True)
    extra_skills = st.text_input(
        "Skills to simulate adding",
        placeholder="e.g. machine learning, AWS",
        help="Appends to your profile and re-runs the model.",
    )
    run_whatif = st.button("⚡ Run what-if", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    if run:
        result = predict_job_risk(skills, education, experience, location, industry)
        st.session_state["last_job_risk"] = result
        st.session_state["last_job_risk_inputs"] = {
            "skills": skills,
            "education": education,
            "experience": experience,
            "location": location,
            "industry": industry,
        }

    if run_whatif:
        inp = st.session_state.get("last_job_risk_inputs")
        if inp:
            base_r, new_r, delta = what_if_improve_skills(
                inp["skills"],
                inp["education"],
                int(inp["experience"]),
                inp["location"],
                inp["industry"],
                extra_skills,
            )
            st.session_state["whatif"] = (base_r, new_r, delta)
        else:
            st.warning("Run **Estimate risk** first, then try what-if.")

    res = st.session_state.get("last_job_risk")
    if not res:
        st.info("Fill the form and click **Estimate risk**.")
    else:
        level_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        color = level_colors.get(res.risk_level, "#94a3b8")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Risk level", res.risk_level)
        with m2:
            st.metric("P(high risk)", f"{res.high_risk_probability_pct}%")
        with m3:
            st.markdown(
                f"<div style='margin-top:1.2rem;'><span style='color:{color};font-weight:700;'>"
                f"●</span> <span style='color:#94a3b8;font-size:0.85rem;'>"
                "High = modeled probability of being in a high-displacement-risk bucket</span></div>",
                unsafe_allow_html=True,
            )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=min(100, max(0, res.high_risk_probability_pct)),
                number={"suffix": "%", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                    "bar": {"color": color},
                    "bgcolor": "rgba(255,255,255,0.04)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16,185,129,0.15)"},
                        {"range": [35, 62], "color": "rgba(245,158,11,0.12)"},
                        {"range": [62, 100], "color": "rgba(239,68,68,0.12)"},
                    ],
                },
                title={"text": "High-risk probability", "font": {"color": "#94a3b8", "size": 14}},
            )
        )
        gauge.update_layout(**plotly_dark_layout(height=220))
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("**Engineered features**")
        feat = res.features
        fv = st.columns(len(FEATURE_NAMES))
        labels = {
            "skill_demand_score": "Skill demand",
            "industry_growth": "Industry growth idx",
            "experience_years": "Experience (yrs)",
            "education_level": "Education (0–4)",
            "location_risk_tier": "Location tier",
        }
        values = [
            feat["skill_demand_score"],
            feat["industry_growth"],
            feat["experience_years"],
            feat["education_level"],
            feat["location_risk_tier"],
        ]
        for i, name in enumerate(FEATURE_NAMES):
            with fv[i]:
                st.caption(labels[name])
                st.write(f"{values[i]:.2f}" if i < 2 else f"{values[i]:.1f}")

        if feat.get("matched_high_demand"):
            st.success("Matched in-demand keywords: **" + "**, **".join(feat["matched_high_demand"][:12]) + "**")
        elif feat.get("parsed_skills"):
            st.caption("No lexicon match — score inferred from profile breadth.")

        st.markdown("**Why this score**")
        for r in res.reasons:
            st.markdown(f"- {r}")
        st.markdown("**Suggestions**")
        for s in res.suggestions:
            st.markdown(f"- {s}")

    wf = st.session_state.get("whatif")
    if wf:
        base_r, new_r, delta = wf
        st.markdown("---")
        st.markdown("**What-if outcome**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Before", f"{base_r.high_risk_probability_pct}%")
        c2.metric("After", f"{new_r.high_risk_probability_pct}%")
        c3.metric("Change", f"{delta:+.1f} pp", delta_color="inverse")
        st.caption("Run **Estimate risk** first, then **Run what-if** to compare.")

    st.markdown("</div>", unsafe_allow_html=True)
