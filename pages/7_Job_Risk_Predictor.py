"""
Page 7 — AI-Based Job Risk Predictor (Feature 6)

User profile → ML risk estimate, feature contribution chart,
industry comparison, what-if skill simulation, and export report.
Runs entirely in-process (no FastAPI required).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.job_risk_model import (
    EDUCATION_LEVELS,
    FEATURE_NAMES,
    INDUSTRY_GROWTH,
    LOCATION_OPTIONS,
    predict_job_risk,
    what_if_improve_skills,
    industry_risk_comparison,
)
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout

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
    st.page_link("pages/10_Skill_Obsolescence.py", label="⚡ Skill Obsolescence")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">🎯 AI Job Risk Predictor</div>
    <div class="hero-subtitle">
        Estimate unemployment-risk probability from skills, education, experience, industry, and location —
        with feature contributions, industry comparison, and what-if skill upgrades.
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <div style="font-size:1.3rem; flex-shrink:0;">⚠️</div>
    <div>
        <div style="font-size:0.82rem; font-weight:700; color:#fbbf24; text-transform:uppercase;
                    letter-spacing:1px; margin-bottom:0.3rem;">Model Disclaimer</div>
        <div style="font-size:0.85rem; color:#94a3b8; line-height:1.55;">
            This predictor is trained on <strong style="color:#e2e8f0;">synthetic data</strong> generated
            from economic heuristics — not real employment records or survey data.
            Risk scores are <em>illustrative estimates</em> for scenario planning only.
            Do not use these results for actual hiring, termination, or career decisions.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Profile Form ─────────────────────────────────────────────────────────────
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


# ─── Results Panel ─────────────────────────────────────────────────────────────
with col_out:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    if run:
        result = predict_job_risk(skills, education, experience, location, industry)
        st.session_state["last_job_risk"] = result
        st.session_state["last_job_risk_inputs"] = {
            "skills": skills, "education": education,
            "experience": experience, "location": location, "industry": industry,
        }

    if run_whatif:
        inp = st.session_state.get("last_job_risk_inputs")
        if inp:
            base_r, new_r, delta = what_if_improve_skills(
                inp["skills"], inp["education"], int(inp["experience"]),
                inp["location"], inp["industry"], extra_skills,
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

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(100, max(0, res.high_risk_probability_pct)),
            number={"suffix": "%", "font": {"color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35], "color": "rgba(16,185,129,0.15)"},
                    {"range": [35, 62], "color": "rgba(245,158,11,0.12)"},
                    {"range": [62, 100],"color": "rgba(239,68,68,0.12)"},
                ],
            },
            title={"text": "High-risk probability", "font": {"color": "#94a3b8", "size": 14}},
        ))
        gauge.update_layout(**plotly_dark_layout(height=220))
        st.plotly_chart(gauge, width='stretch')

        st.markdown("**Engineered features**")
        feat = res.features
        fv = st.columns(len(FEATURE_NAMES))
        labels = {
            "skill_demand_score": "Skill demand",
            "industry_growth": "Industry growth",
            "experience_years": "Experience (yrs)",
            "education_level": "Education (0–4)",
            "location_risk_tier": "Location tier",
        }
        values = [
            feat["skill_demand_score"], feat["industry_growth"],
            feat["experience_years"], feat["education_level"], feat["location_risk_tier"],
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

        if new_r.contributions:
            # Show how contributions shifted
            labels_map = {
                "skill_demand_score": "Skill demand",
                "industry_growth":    "Industry growth",
                "experience_years":   "Experience",
                "education_level":    "Education",
                "location_risk_tier": "Location",
            }
            base_c = base_r.contributions or {}
            new_c  = new_r.contributions  or {}
            feat_labels = [labels_map.get(k, k) for k in FEATURE_NAMES]
            delta_vals  = [round(new_c.get(k, 0) - base_c.get(k, 0), 4) for k in FEATURE_NAMES]
            bar_colors  = ["#10b981" if v < 0 else "#ef4444" for v in delta_vals]
            fig_wi = go.Figure(go.Bar(
                x=feat_labels, y=delta_vals,
                marker_color=bar_colors,
                hovertemplate="%{x}: Δ%{y:.4f}<extra></extra>",
            ))
            fig_wi.add_hline(y=0, line=dict(color="#64748b", width=1, dash="dash"))
            fig_wi.update_layout(
                **plotly_dark_layout(height=200, showlegend=False),
                title=dict(text="Contribution shift after skill upgrade", font=dict(color="#94a3b8", size=13)),
                yaxis_title="Change in log-odds contribution",
            )
            st.plotly_chart(fig_wi, width='stretch')

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Feature Contribution Chart ───────────────────────────────────────────────
res = st.session_state.get("last_job_risk")
if res and res.contributions:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧩 FEATURE CONTRIBUTION BREAKDOWN</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="color:#64748b; font-size:0.88rem; margin-bottom:1.2rem;">
How much each feature pushes your risk above (<span style="color:#f87171;">red</span>) or below
(<span style="color:#34d399;">green</span>) the average profile. Based on logistic regression coefficients.
</p>""", unsafe_allow_html=True)

    labels_map = {
        "skill_demand_score": "Skill Demand Score",
        "industry_growth":    "Industry Growth Index",
        "experience_years":   "Years of Experience",
        "education_level":    "Education Level",
        "location_risk_tier": "Location Risk Tier",
    }
    contrib_items = sorted(res.contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    feat_labels = [labels_map.get(k, k) for k, _ in contrib_items]
    contrib_vals = [v for _, v in contrib_items]
    bar_colors   = ["#10b981" if v < 0 else "#ef4444" for v in contrib_vals]

    fig_contrib = go.Figure(go.Bar(
        x=contrib_vals,
        y=feat_labels,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        text=[f"{v:+.3f}" for v in contrib_vals],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_contrib.add_vline(x=0, line=dict(color="#64748b", width=1.5, dash="dash"))
    fig_contrib.update_layout(
        **plotly_dark_layout(height=300, showlegend=False),
        xaxis_title="Log-odds contribution (negative = lowers risk, positive = raises risk)",
        margin=dict(l=10, r=60, t=20, b=30),
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_contrib, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Industry Comparison ───────────────────────────────────────────────────────
inp = st.session_state.get("last_job_risk_inputs")
if inp:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏭 INDUSTRY RISK COMPARISON</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="color:#64748b; font-size:0.88rem; margin-bottom:1.2rem;">
Your current skills, education, and location — but across all industries. Sorted best to worst.
</p>""", unsafe_allow_html=True)

    ind_rows = industry_risk_comparison(
        inp["skills"], inp["education"], int(inp["experience"]), inp["location"]
    )
    ind_df = pd.DataFrame(ind_rows)

    bar_clr = []
    for lvl in ind_df["Level"]:
        if lvl == "Low":
            bar_clr.append("#10b981")
        elif lvl == "Medium":
            bar_clr.append("#f59e0b")
        else:
            bar_clr.append("#ef4444")

    fig_ind = go.Figure(go.Bar(
        x=ind_df["Risk (%)"],
        y=ind_df["Industry"],
        orientation="h",
        marker_color=bar_clr,
        text=[f"{v}%" for v in ind_df["Risk (%)"]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="%{y}: %{x}%<extra></extra>",
    ))

    # Mark current industry
    current_ind = inp["industry"]
    if current_ind in ind_df["Industry"].values:
        idx = ind_df[ind_df["Industry"] == current_ind].index[0]
        y_pos = int(ind_df[ind_df["Industry"] == current_ind].index[0])
        fig_ind.add_annotation(
            x=ind_df.loc[idx, "Risk (%)"] + 2,
            y=current_ind,
            text="◀ current",
            showarrow=False,
            font=dict(color="#818cf8", size=11),
        )

    fig_ind.update_layout(
        **plotly_dark_layout(height=350, showlegend=False),
        xaxis_title="Estimated Risk (%)",
        xaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=80, t=20, b=30),
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_ind, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Export Report ─────────────────────────────────────────────────────────────
if res and inp:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 EXPORT RISK REPORT</div>', unsafe_allow_html=True)

    def _build_risk_report() -> bytes:
        lines = [
            "UNEMPLOYMENT INTELLIGENCE PLATFORM — JOB RISK ASSESSMENT",
            "=" * 60,
            "",
            "PROFILE",
            "-" * 40,
            f"  Skills       : {inp['skills'] or '(none provided)'}",
            f"  Education    : {inp['education']}",
            f"  Experience   : {inp['experience']} years",
            f"  Industry     : {inp['industry']}",
            f"  Location     : {inp['location']}",
            "",
            "RISK RESULT",
            "-" * 40,
            f"  Risk Level        : {res.risk_level}",
            f"  High-Risk Prob.   : {res.high_risk_probability_pct}%",
            "",
            "FEATURE VALUES",
            "-" * 40,
            f"  Skill Demand Score  : {res.features.get('skill_demand_score', 'N/A'):.2f}",
            f"  Industry Growth Idx : {res.features.get('industry_growth', 'N/A'):.2f}",
            f"  Experience (yrs)    : {res.features.get('experience_years', 'N/A'):.0f}",
            f"  Education (0–4)     : {res.features.get('education_level', 'N/A'):.0f}",
            f"  Location Tier       : {res.features.get('location_risk_tier', 'N/A'):.0f}",
            "",
            "FEATURE CONTRIBUTIONS (log-odds, negative = lowers risk)",
            "-" * 40,
        ]
        if res.contributions:
            for k, v in sorted(res.contributions.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"  {k:<24}: {v:+.4f}")
        lines += ["", "WHY THIS SCORE", "-" * 40]
        for r in res.reasons:
            lines.append(f"  - {r}")
        lines += ["", "SUGGESTIONS", "-" * 40]
        for s in res.suggestions:
            lines.append(f"  - {s}")
        wf = st.session_state.get("whatif")
        if wf:
            base_r, new_r, delta = wf
            lines += [
                "", "WHAT-IF: SKILL UPGRADE", "-" * 40,
                f"  Before : {base_r.high_risk_probability_pct}%",
                f"  After  : {new_r.high_risk_probability_pct}%",
                f"  Change : {delta:+.1f} pp",
            ]
        lines += ["", "Generated by Unemployment Intelligence Platform"]
        return "\n".join(lines).encode("utf-8")

    st.download_button(
        label="⬇️ Download Risk Assessment Report (.txt)",
        data=_build_risk_report(),
        file_name="uip_job_risk_report.txt",
        mime="text/plain",
    )
