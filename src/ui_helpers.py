"""
ui_helpers.py
Shared UI utilities: CSS, KPI cards, badges, Plotly layout, and animated counters.
"""

API_BASE_URL = "http://localhost:8000"

DARK_CSS = """
<style>
/* ── Base ───────────────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0a0f1e !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] {
    background: rgba(15,20,40,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Glass cards ────────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* ── KPI card ───────────────────────────────────────────────────────────────── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    border-radius: 16px 16px 0 0;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b !important;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9 !important;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-delta-up     { font-size: 0.78rem; color: #ef4444 !important; font-weight: 600; }
.kpi-delta-down   { font-size: 0.78rem; color: #10b981 !important; font-weight: 600; }
.kpi-delta-neutral{ font-size: 0.78rem; color: #94a3b8 !important; font-weight: 600; }

/* ── Animated KPI counter ───────────────────────────────────────────────────── */
@keyframes countUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.kpi-animated { animation: countUp 0.6s ease-out both; }

/* ── Section title ──────────────────────────────────────────────────────────── */
.section-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6366f1 !important;
    margin-bottom: 1rem;
}

/* ── Page hero ──────────────────────────────────────────────────────────────── */
.page-hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9 !important;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #64748b !important;
    line-height: 1.6;
}

/* ── Badges (canonical names) ───────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
/* Canonical colour names */
.badge-blue    { background: rgba(99,102,241,0.15);  color: #818cf8 !important; border: 1px solid rgba(99,102,241,0.25); }
.badge-green   { background: rgba(16,185,129,0.12);  color: #34d399 !important; border: 1px solid rgba(16,185,129,0.2); }
.badge-red     { background: rgba(239,68,68,0.12);   color: #f87171 !important; border: 1px solid rgba(239,68,68,0.2); }
.badge-yellow  { background: rgba(245,158,11,0.12);  color: #fbbf24 !important; border: 1px solid rgba(245,158,11,0.2); }
.badge-purple  { background: rgba(139,92,246,0.12);  color: #a78bfa !important; border: 1px solid rgba(139,92,246,0.2); }

/* Semantic aliases — used by pages that pass danger/success/info/warning */
.badge-success { background: rgba(16,185,129,0.12);  color: #34d399 !important; border: 1px solid rgba(16,185,129,0.2); }
.badge-danger  { background: rgba(239,68,68,0.12);   color: #f87171 !important; border: 1px solid rgba(239,68,68,0.2); }
.badge-info    { background: rgba(99,102,241,0.15);  color: #818cf8 !important; border: 1px solid rgba(99,102,241,0.25); }
.badge-warning { background: rgba(245,158,11,0.12);  color: #fbbf24 !important; border: 1px solid rgba(245,158,11,0.2); }

/* ── Skill chip ─────────────────────────────────────────────────────────────── */
.skill-chip {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    color: #818cf8;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}

/* ── Timeline ───────────────────────────────────────────────────────────────── */
.timeline-item {
    display: flex;
    gap: 1rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.timeline-content { flex: 1; }
.timeline-year {
    font-size: 0.85rem;
    font-weight: 800;
    color: #6366f1;
}
.timeline-desc {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* ── Disclaimer banner ──────────────────────────────────────────────────────── */
.disclaimer-banner {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
}

/* ── Streamlit overrides ─────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}
[data-testid="stDataFrame"] table {
    background: rgba(255,255,255,0.02) !important;
    color: #e2e8f0 !important;
}
.stSlider > div > div > div { background: #6366f1 !important; }

/* ── Data source pill ───────────────────────────────────────────────────────── */
.data-source-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #34d399;
}
</style>
"""

# ---------------------------------------------------------------------------
# Animated KPI counter (CSS-only, renders instantly on load)
# ---------------------------------------------------------------------------
COUNTER_JS = """
<script>
(function(){
    const els = document.querySelectorAll('.kpi-value[data-target]');
    els.forEach(el => {
        const target = parseFloat(el.dataset.target);
        const suffix = el.dataset.suffix || '';
        const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
        let start = 0;
        const duration = 800;
        const step = (timestamp) => {
            if (!start) start = timestamp;
            const progress = Math.min((timestamp - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = (target * eased).toFixed(decimals) + suffix;
            if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    });
})();
</script>
"""


def render_kpi_card(
    icon: str,
    label: str,
    value: str,
    subtitle: str = "",
    delta_type: str = "neutral",
    animate: bool = False,
) -> str:
    delta_class = f"kpi-delta-{delta_type}"
    subtitle_html = f'<div class="{delta_class}">{subtitle}</div>' if subtitle else ""
    anim_class = " kpi-animated" if animate else ""
    return f"""
<div class="kpi-card{anim_class}">
    <div class="kpi-icon">{icon}</div>
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    {subtitle_html}
</div>
"""


def render_badge(text: str, color: str = "blue") -> str:
    """
    color accepts both canonical names (blue/green/red/yellow/purple)
    and semantic aliases (success/danger/info/warning).
    """
    return f'<span class="badge badge-{color}">{text}</span>'


def render_data_source(label: str) -> str:
    return f'<div class="data-source-pill">{label}</div>'


def plotly_dark_layout(height: int = 400, showlegend: bool = True, **kwargs) -> dict:
    base = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=showlegend,
        font=dict(color="#94a3b8", family="Inter, system-ui, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.03)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color="#94a3b8"),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#64748b"),
        ),
    )
    base.update(kwargs)
    return base
