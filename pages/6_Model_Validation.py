"""
Page 6 — Model Validation (Feature 5)
Gauge chart for R², backtest actual vs predicted, residuals analysis,
year-by-year error table, and export metrics report.
"""
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
    st.page_link("pages/10_Skill_Obsolescence.py", label="⚡ Skill Obsolescence")

st.markdown("""
<div class="page-hero">
    <div class="hero-title">🔬 Model Validation Lab</div>
    <div class="hero-subtitle">Backtest accuracy, R² score, error metrics, residuals analysis, and model reliability assessment</div>
</div>""", unsafe_allow_html=True)


@st.cache_data(ttl=120)
def get_validation():
    try:
        r = requests.get(f"{API_BASE_URL}/validate", timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=60)
def get_backtest(years):
    try:
        r = requests.post(f"{API_BASE_URL}/backtest", json={"test_years": years}, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


val_data = get_validation()
bt_data  = get_backtest(test_years)

if not val_data and not bt_data:
    st.error("⚠️ Cannot connect to API. Make sure the FastAPI Backend workflow is running.")
    st.stop()


# ─── Validation KPIs ──────────────────────────────────────────────────────────
mae  = val_data.get("mae",  "N/A") if val_data else "N/A"
mape = val_data.get("mape", "N/A") if val_data else "N/A"
rmse = val_data.get("rmse", "N/A") if val_data else "N/A"
r2   = val_data.get("r2",   None)  if val_data else None
da   = val_data.get("directional_accuracy", "N/A") if val_data else "N/A"
fb   = val_data.get("forecast_bias", "N/A") if val_data else "N/A"

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


# ─── R² Gauge + Reliability Metrics ───────────────────────────────────────────
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
                dict(range=[0,  50],  color="rgba(239,68,68,0.12)"),
                dict(range=[50, 70],  color="rgba(245,158,11,0.12)"),
                dict(range=[70, 85],  color="rgba(6,182,212,0.12)"),
                dict(range=[85, 100], color="rgba(16,185,129,0.12)"),
            ],
            threshold=dict(line=dict(color="#f59e0b", width=2), thickness=0.6, value=70),
        ),
        title=dict(text="R² Score (Target: 70%+)", font=dict(color="#94a3b8", size=14)),
    ))
    fig_gauge.update_layout(
        **plotly_dark_layout(height=300, margin=dict(l=20, r=20, t=60, b=10)),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(f"""
    <div style="text-align:center; margin-top:0.5rem;">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:1px; color:#64748b; margin-bottom:0.5rem;">MODEL GRADE</div>
        <div style="display:inline-block; padding:0.5rem 2rem;
                    background:{gcolor}22; border:1px solid {gcolor}44; border-radius:12px;">
            <span style="font-size:1.3rem; font-weight:800; color:{gcolor};">{grade}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_metrics:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Full Validation Report</div>', unsafe_allow_html=True)

    metrics_rows = [
        ("Mean Absolute Error (MAE)",    f"{mae:.4f}"  if isinstance(mae,  (int, float)) else str(mae)),
        ("Mean Abs. % Error (MAPE)",     f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)),
        ("Root Mean Sq. Error (RMSE)",   f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)),
        ("R² Variance Explained",        f"{r2:.4f}"   if r2 is not None else "N/A"),
        ("Directional Accuracy",         f"{da:.1f}%"  if isinstance(da,   (int, float)) else str(da)),
        ("Forecast Bias",                f"{fb:.4f}"   if isinstance(fb,   (int, float)) else str(fb)),
    ]
    for label, val in metrics_rows:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.7rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:#94a3b8; font-size:0.88rem;">{label}</span>
            <span style="color:#e2e8f0; font-weight:700; font-size:0.92rem; font-family:monospace;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if r2 is not None:
        if r2 > 0.7:
            st.success("✅ Model demonstrates strong predictive performance. Suitable for decision support.")
        elif r2 > 0.5:
            st.warning("⚠️ Model shows moderate performance. Treat forecasts as indicative, not definitive.")
        else:
            st.error("❌ Model accuracy is limited. Additional calibration required.")
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Backtest Chart ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Backtest: Actual vs Predicted</div>', unsafe_allow_html=True)

hist_df = pd.DataFrame()
bt_df   = pd.DataFrame()

if bt_data:
    bt_mae  = bt_data.get("mae")
    bt_mape = bt_data.get("mape")
    hist_raw = bt_data.get("historical", [])
    bt_raw   = bt_data.get("backtest",   [])

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

        merged_bt = pd.merge(
            hist_df[["Year", "Unemployment_Rate"]],
            bt_df[["Year",  "Predicted_Unemployment"]],
            on="Year", how="inner",
        )
        if not merged_bt.empty:
            fig_bt.add_trace(go.Scatter(
                x=merged_bt["Year"].tolist() + merged_bt["Year"].tolist()[::-1],
                y=merged_bt["Predicted_Unemployment"].tolist() + merged_bt["Unemployment_Rate"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(99,102,241,0.06)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Error Band",
                hoverinfo="skip",
            ))

        fig_bt.update_layout(**plotly_dark_layout(height=380))
        fig_bt.update_layout(
            xaxis_title="Year",
            yaxis_title="Unemployment Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.info("No backtest comparison data available.")
else:
    st.warning("Backtest data not available. Check API connection.")

st.markdown("</div>", unsafe_allow_html=True)


# ─── Residuals Analysis ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 RESIDUALS ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#64748b; font-size:0.88rem; margin-bottom:1.2rem;">
Residuals (Predicted − Actual) per year. Values near zero are ideal.
Consistent positive bias = over-prediction; negative = under-prediction.
</p>""", unsafe_allow_html=True)

detail = val_data.get("detail", []) if val_data else []
if detail:
    detail_df = pd.DataFrame(detail)
    detail_df["Residual"] = (detail_df["Predicted"] - detail_df["Actual"]).round(4)
    detail_df["Abs Error"] = detail_df["Residual"].abs().round(4)

    col_res, col_hist = st.columns([3, 2])

    with col_res:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📉 Residuals Over Time</div>', unsafe_allow_html=True)

        bar_colors = ["#ef4444" if v > 0 else "#10b981" for v in detail_df["Residual"]]
        fig_res = go.Figure()
        fig_res.add_hline(y=0, line=dict(color="#64748b", width=1, dash="dash"))
        fig_res.add_trace(go.Bar(
            x=detail_df["Year"],
            y=detail_df["Residual"],
            marker_color=bar_colors,
            name="Residual",
            hovertemplate="Year: %{x}<br>Residual: %{y:.4f}<extra></extra>",
        ))
        fig_res.update_layout(**plotly_dark_layout(height=300, showlegend=False))
        fig_res.update_layout(
            xaxis_title="Year",
            yaxis_title="Predicted − Actual (pp)",
        )
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_hist:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Error Distribution</div>', unsafe_allow_html=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=detail_df["Residual"],
            nbinsx=8,
            marker_color="rgba(99,102,241,0.6)",
            marker_line=dict(color="rgba(99,102,241,1)", width=1),
            name="Residual Distribution",
        ))
        fig_hist.add_vline(x=0, line=dict(color="#f59e0b", width=2, dash="dash"))
        fig_hist.update_layout(**plotly_dark_layout(height=300, showlegend=False))
        fig_hist.update_layout(xaxis_title="Residual Value", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Year-by-year error table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Year-by-Year Error Breakdown</div>', unsafe_allow_html=True)

    display_df = detail_df[["Year", "Actual", "Predicted", "Residual", "Abs Error"]].copy()
    display_df.columns = ["Year", "Actual (%)", "Predicted (%)", "Residual (pp)", "Abs Error (pp)"]

    def _style_residual(val):
        try:
            v = float(val)
            if abs(v) < 0.1:
                return "color: #34d399; font-weight: 700;"
            elif abs(v) < 0.3:
                return "color: #fbbf24; font-weight: 700;"
            else:
                return "color: #f87171; font-weight: 700;"
        except Exception:
            return ""

    st.dataframe(
        display_df.style.applymap(_style_residual, subset=["Residual (pp)", "Abs Error (pp)"]),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Detailed residuals require validation data from the API.")


# ─── Export Metrics Report ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📥 EXPORT VALIDATION REPORT</div>', unsafe_allow_html=True)


def _build_val_report() -> bytes:
    lines = [
        "UNEMPLOYMENT INTELLIGENCE PLATFORM — MODEL VALIDATION REPORT",
        "=" * 60,
        "",
        "VALIDATION METRICS (60% train / 40% test split)",
        "-" * 40,
        f"  MAE                  : {mae}",
        f"  MAPE                 : {mape}",
        f"  RMSE                 : {rmse}",
        f"  R² Score             : {r2}",
        f"  Directional Accuracy : {da}",
        f"  Forecast Bias        : {fb}",
        f"  Model Grade          : {grade}",
        "",
    ]
    if bt_data:
        lines += [
            f"BACKTEST METRICS ({test_years}-year holdout)",
            "-" * 40,
            f"  Backtest MAE  : {bt_data.get('mae', 'N/A')}",
            f"  Backtest MAPE : {bt_data.get('mape', 'N/A')}",
            "",
        ]
    if detail:
        lines += ["YEAR-BY-YEAR ERROR BREAKDOWN", "-" * 40]
        lines.append(f"  {'Year':>6}  {'Actual':>8}  {'Predicted':>10}  {'Residual':>10}")
        for row in detail:
            actual_v    = row.get("Actual", "N/A")
            predicted_v = row.get("Predicted", "N/A")
            try:
                residual = round(float(predicted_v) - float(actual_v), 4)
            except Exception:
                residual = "N/A"
            lines.append(f"  {row.get('Year','?'):>6}  {actual_v:>8}  {predicted_v:>10}  {residual:>10}")
    lines += ["", "Generated by Unemployment Intelligence Platform"]
    return "\n".join(str(l) for l in lines).encode("utf-8")


st.download_button(
    label="⬇️ Download Validation Report (.txt)",
    data=_build_val_report(),
    file_name="uip_model_validation_report.txt",
    mime="text/plain",
)
