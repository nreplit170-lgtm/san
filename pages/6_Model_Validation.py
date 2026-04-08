"""
Page 6 — Model Validation
Gauge chart for R², backtest actual vs predicted, metrics table, and reliability badge.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from src.ui_helpers import DARK_CSS, render_kpi_card, render_badge, plotly_dark_layout, API_BASE_URL

st.set_page_config(page_title="Model Validation | UIP", page_icon="🔬", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 Model Validation")
    test_years = st.slider("Backtest Test Years", 2, 8, 5)
    st.markdown("---")
    st.markdown("**🌐 Navigation**")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Overview.py", label="📊 Overview")
    st.page_link("pages/2_Simulator.py", label="🧪 Simulator")
    st.page_link("pages/3_Sector_Analysis.py", label="🏭 Sector Analysis")
    st.page_link("pages/4_Career_Lab.py", label="💼 Career Lab")
    st.page_link("pages/5_AI_Insights.py", label="🤖 AI Insights")
    st.page_link("pages/7_Job_Risk_Predictor.py", label="🎯 Job Risk (AI)")
    st.page_link("pages/8_Job_Market_Pulse.py", label="📡 Market Pulse")
    st.page_link("pages/9_Geo_Career_Advisor.py", label="🗺️ Geo Career")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">🔬 Model Validation Lab</div>
    <div class="hero-subtitle">Backtest accuracy, R² score, error metrics, and model reliability assessment</div>
</div>""", unsafe_allow_html=True)

@st.cache_data(ttl=120)
def get_validation():
    try:
        r = requests.get(f"{API_BASE_URL}/validate", timeout=20)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

@st.cache_data(ttl=60)
def get_backtest(years):
    try:
        r = requests.post(f"{API_BASE_URL}/backtest", json={"test_years": years}, timeout=20)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

val_data = get_validation()
bt_data  = get_backtest(test_years)

if not val_data and not bt_data:
    st.error("⚠️ Cannot connect to API. Start: `uvicorn src.api:app --reload`")
    st.stop()

# ─── Validation KPIs ──────────────────────────────────────────────────────────
mae  = val_data.get("mae",  "N/A") if val_data else "N/A"
mape = val_data.get("mape", "N/A") if val_data else "N/A"
rmse = val_data.get("rmse", "N/A") if val_data else "N/A"
r2   = val_data.get("r2_score", None) if val_data else None
da   = val_data.get("directional_accuracy", "N/A") if val_data else "N/A"
fb   = val_data.get("forecast_bias", "N/A") if val_data else "N/A"

# reliability grade
if r2 is not None:
    if r2 > 0.8:
        grade, gcolor, gkind = "A  — Excellent", "#10b981", "success"
    elif r2 > 0.65:
        grade, gcolor, gkind = "B  — Good", "#06b6d4", "info"
    elif r2 > 0.5:
        grade, gcolor, gkind = "C  — Moderate", "#f59e0b", "warning"
    else:
        grade, gcolor, gkind = "D  — Needs Improvement", "#ef4444", "danger"
else:
    grade, gcolor, gkind = "N/A", "#64748b", "info"

c1, c2, c3, c4 = st.columns(4)
with c1:
    mae_s = f"{mae:.3f}" if isinstance(mae, (int, float)) else str(mae)
    st.markdown(render_kpi_card("📏", "MAE", mae_s, delta_type="neutral"), unsafe_allow_html=True)
with c2:
    mape_s = f"{mape:.1f}%" if isinstance(mape, (int, float)) else str(mape)
    st.markdown(render_kpi_card("📐", "MAPE", mape_s, delta_type="neutral"), unsafe_allow_html=True)
with c3:
    rmse_s = f"{rmse:.3f}" if isinstance(rmse, (int, float)) else str(rmse)
    st.markdown(render_kpi_card("📎", "RMSE", rmse_s, delta_type="neutral"), unsafe_allow_html=True)
with c4:
    r2_s = f"{r2:.4f}" if r2 is not None else "N/A"
    dt = "down" if r2 is not None and r2 > 0.6 else "up"
    st.markdown(render_kpi_card("🏆", "R² Score", r2_s, grade, dt), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── R² Gauge + Reliability Score ─────────────────────────────────────────────
col_gauge, col_metrics = st.columns([1, 1])

with col_gauge:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Model Reliability Gauge</div>', unsafe_allow_html=True)

    r2_val = r2 if r2 is not None else 0.5
    r2_pct = round(r2_val * 100, 1)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=r2_pct,
        number=dict(suffix="%", font=dict(color="#e2e8f0", size=36)),
        delta=dict(reference=70, increasing=dict(color="#10b981"), decreasing=dict(color="#ef4444")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#334155",
                      tickfont=dict(color="#64748b")),
            bar=dict(color="#6366f1", thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.05)",
            steps=[
                dict(range=[0, 50],  color="rgba(239,68,68,0.12)"),
                dict(range=[50, 70], color="rgba(245,158,11,0.12)"),
                dict(range=[70, 85], color="rgba(6,182,212,0.12)"),
                dict(range=[85, 100],color="rgba(16,185,129,0.12)"),
            ],
            threshold=dict(line=dict(color="#f59e0b", width=2), thickness=0.6, value=70),
        ),
        title=dict(text="R² Score (Target: 70%+)", font=dict(color="#94a3b8", size=14)),
    ))
    fig_gauge.update_layout(
        **plotly_dark_layout(height=300),
        margin=dict(l=20, r=20, t=60, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Reliability badge
    st.markdown(f"""
    <div style="text-align:center; margin-top:0.5rem;">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:1px; color:#64748b; margin-bottom:0.5rem;">MODEL GRADE</div>
        <div style="display:inline-block; padding:0.5rem 2rem;
                    background:rgba({','.join(['103','105','190'] if gkind=='primary' else ['16','185','129'] if gkind=='success' else ['6','182','212'] if gkind=='info' else ['245','158','11'] if gkind=='warning' else ['239','68','68'])},0.15);
                    border:1px solid {gcolor}40; border-radius:12px;">
            <span style="font-size:1.3rem; font-weight:800; color:{gcolor};">{grade}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_metrics:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Full Validation Report</div>', unsafe_allow_html=True)

    metrics_rows = [
        ("Mean Absolute Error (MAE)", f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)),
        ("Mean Abs. % Error (MAPE)", f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)),
        ("Root Mean Sq. Error (RMSE)", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)),
        ("R² Variance Explained", f"{r2:.4f}" if r2 is not None else "N/A"),
        ("Directional Accuracy", f"{da:.1f}%" if isinstance(da, (int, float)) else str(da)),
        ("Forecast Bias", f"{fb:.4f}" if isinstance(fb, (int, float)) else str(fb)),
    ]
    for label, val in metrics_rows:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.7rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:#94a3b8; font-size:0.88rem;">{label}</span>
            <span style="color:#e2e8f0; font-weight:700; font-size:0.92rem; font-family:monospace;">{val}</span>
        </div>""", unsafe_allow_html=True)

    # overall assessment
    st.markdown("<br>", unsafe_allow_html=True)
    if r2 is not None:
        if r2 > 0.7:
            st.success("✅ Model demonstrates strong predictive performance. Suitable for decision support.")
        elif r2 > 0.5:
            st.warning("⚠️ Model shows moderate performance. Treat forecasts as indicative, not definitive.")
        else:
            st.error("❌ Model accuracy is limited. Additional calibration required.")
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Backtest Chart ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Backtest: Actual vs Predicted</div>', unsafe_allow_html=True)

if bt_data:
    bt_mae  = bt_data.get("mae")
    bt_mape = bt_data.get("mape")
    hist_raw = bt_data.get("historical", [])
    bt_raw   = bt_data.get("backtest", [])

    col_bm1, col_bm2 = st.columns(2)
    with col_bm1:
        s = f"{bt_mae:.3f}" if isinstance(bt_mae, (int, float)) else "N/A"
        st.markdown(render_kpi_card("📏", f"Backtest MAE ({test_years} yrs)", s, delta_type="neutral"), unsafe_allow_html=True)
    with col_bm2:
        s = f"{bt_mape:.2f}%" if isinstance(bt_mape, (int, float)) else "N/A"
        st.markdown(render_kpi_card("📐", "Backtest MAPE", s, delta_type="neutral"), unsafe_allow_html=True)

    if hist_raw and bt_raw:
        hist_df = pd.DataFrame(hist_raw)
        bt_df   = pd.DataFrame(bt_raw)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=hist_df["Year"], y=hist_df["Unemployment_Rate"],
            mode="lines+markers", name="Actual",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8, color="#f87171", symbol="circle"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_df["Year"], y=bt_df["Predicted_Unemployment"],
            mode="lines+markers", name="Predicted",
            line=dict(color="#10b981", width=3, dash="dash"),
            marker=dict(size=8, color="#34d399", symbol="diamond"),
        ))

        # Error shading
        merged = pd.merge(hist_df[["Year","Unemployment_Rate"]], bt_df[["Year","Predicted_Unemployment"]], on="Year", how="inner")
        if not merged.empty:
            fig_bt.add_trace(go.Scatter(
                x=merged["Year"].tolist() + merged["Year"].tolist()[::-1],
                y=merged["Predicted_Unemployment"].tolist() + merged["Unemployment_Rate"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(99,102,241,0.06)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Error Band",
                hoverinfo="skip",
                showlegend=True,
            ))

        fig_bt.update_layout(**plotly_dark_layout(height=380))
        fig_bt.update_layout(
            xaxis_title="Year",
            yaxis_title="Unemployment Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.info("No backtest comparison data available")
else:
    st.warning("Backtest data not available. Check API connection.")

st.markdown("</div>", unsafe_allow_html=True)
