"""
historical_events.py
Curated list of major economic events that affected Indian unemployment.
Used to annotate forecast and historical charts with context.
"""

INDIA_ECONOMIC_EVENTS = [
    {
        "year": 1991,
        "label": "1991 Balance of Payments Crisis",
        "short": "BoP Crisis",
        "description": "India's foreign exchange reserves fell to near zero, triggering an IMF bailout and sweeping liberalisation reforms.",
        "type": "crisis",
        "color": "#ef4444",
    },
    {
        "year": 1997,
        "label": "1997 Asian Financial Crisis",
        "short": "Asian Crisis",
        "description": "Regional financial contagion from East Asia. India was relatively insulated but experienced capital outflows.",
        "type": "external_shock",
        "color": "#f59e0b",
    },
    {
        "year": 2000,
        "label": "2000 Dot-Com Bust",
        "short": "Dot-Com Bust",
        "description": "Global tech sector collapse. Impact on Indian IT exports and software services employment.",
        "type": "sector_shock",
        "color": "#f59e0b",
    },
    {
        "year": 2008,
        "label": "2008 Global Financial Crisis",
        "short": "GFC 2008",
        "description": "The worst global recession since 1929. India's export-oriented sectors (IT, manufacturing) saw significant job losses.",
        "type": "crisis",
        "color": "#ef4444",
    },
    {
        "year": 2016,
        "label": "2016 Demonetisation",
        "short": "Demonetisation",
        "description": "Overnight ban on ₹500/₹1000 notes disrupted the informal economy, which employs ~90% of India's workforce.",
        "type": "policy_shock",
        "color": "#8b5cf6",
    },
    {
        "year": 2017,
        "label": "2017 GST Implementation",
        "short": "GST Rollout",
        "description": "Introduction of the Goods and Services Tax caused short-term disruption for SMEs and informal businesses.",
        "type": "policy_shock",
        "color": "#8b5cf6",
    },
    {
        "year": 2020,
        "label": "2020 COVID-19 Pandemic",
        "short": "COVID-19",
        "description": "India's strictest lockdown (March–May 2020) caused unemployment to spike to ~23.5% at peak (CMIE data). Tens of millions of migrant workers lost jobs overnight.",
        "type": "crisis",
        "color": "#ef4444",
    },
    {
        "year": 2021,
        "label": "2021 Second COVID Wave",
        "short": "COVID Wave 2",
        "description": "Delta variant surge caused state-level lockdowns and renewed economic disruption, though less severe than 2020.",
        "type": "crisis",
        "color": "#f59e0b",
    },
    {
        "year": 2022,
        "label": "2022 Global Inflation & Rate Hikes",
        "short": "Inflation Shock",
        "description": "Aggressive global monetary tightening to combat inflation slowed growth and raised unemployment pressures.",
        "type": "external_shock",
        "color": "#f59e0b",
    },
]

TYPE_COLORS = {
    "crisis": "#ef4444",
    "external_shock": "#f59e0b",
    "sector_shock": "#06b6d4",
    "policy_shock": "#8b5cf6",
    "recovery": "#10b981",
}


def get_events_in_range(year_min: int, year_max: int) -> list:
    return [e for e in INDIA_ECONOMIC_EVENTS if year_min <= e["year"] <= year_max]


def get_all_events() -> list:
    return INDIA_ECONOMIC_EVENTS
