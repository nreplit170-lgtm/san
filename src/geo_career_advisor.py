"""
Geo-aware career advisor (Feature 3).

Industry-standard building blocks:
  • WGS84 coordinates from a maintained city reference table (CSV).
  • Folium / Leaflet-style maps (consumer of GeoJSON-style point data).
  • Optional geocoding via Nominatim (OpenStreetMap) with required User-Agent
    and caching — suitable for low-volume interactive use; swap for a
    commercial geocoder (Google, Mapbox, HERE) in production.
  • Location quotient–style metrics: local skill mention rate vs national.
  • Relocation “delta” uses the existing calibrated job-risk model twice
    (different location tiers) — not hard-coded percentage claims.

Replace `job_postings_sample.csv` with ATS or job-board exports to drive
production dashboards; reference cities can be extended or loaded from GeoNames.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.job_market_pulse import load_job_postings, phrase_in_blob, skill_phrase_list
from src.job_risk_model import LOCATION_OPTIONS, predict_job_risk

REFERENCE_CSV = Path(__file__).resolve().parent.parent / "data" / "geo" / "india_city_reference.csv"

# Nominatim policy: identify the application; change domain if you ship publicly.
NOMINATIM_USER_AGENT = "unemployment-intelligence-platform/1.0 (career-advisor; open-source demo)"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def normalize_city_key(raw: str) -> str:
    if not raw or not str(raw).strip():
        return ""
    s = str(raw).strip().lower()
    aliases = {
        "bengaluru": "bangalore",
        "blr": "bangalore",
        "hyd": "hyderabad",
        "bombay": "mumbai",
        "gurgaon": "gurugram",
        "ncr": "delhi",
        "new delhi": "delhi",
    }
    return aliases.get(s, s)


@lru_cache(maxsize=1)
def load_city_reference() -> pd.DataFrame:
    path = REFERENCE_CSV
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["city_key"] = df["city_key"].astype(str).str.strip().str.lower()
    return df.drop_duplicates(subset=["city_key"], keep="first")


def resolve_city_row(location_label: str) -> Optional[pd.Series]:
    ref = load_city_reference()
    if ref.empty:
        return None
    key = normalize_city_key(location_label)
    hit = ref.loc[ref["city_key"] == key]
    if hit.empty:
        return None
    return hit.iloc[0]


def geocode_place(query: str) -> Optional[Tuple[float, float, str]]:
    """
    Forward geocode with GeoPy Nominatim. Call sparingly (≈1 req/s per OSM usage policy).
    Returns (lat, lon, display_name) or None.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return None
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError

        geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=10)
        loc = geolocator.geocode(f"{q}, India", country_codes="in", exactly_one=True)
        if loc is None:
            loc = geolocator.geocode(q, exactly_one=True)
        if loc is None:
            return None
        return float(loc.latitude), float(loc.longitude), loc.address
    except (GeocoderTimedOut, GeocoderServiceError, ImportError, OSError):
        return None


def postings_with_city_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "location" not in df.columns:
        return df.iloc[0:0].copy()
    out = df.copy()
    out["city_key"] = out["location"].map(lambda x: normalize_city_key(str(x)))
    return out


def aggregate_city_labour_market(df: pd.DataFrame) -> pd.DataFrame:
    """Posting counts and salary midpoints per normalized city."""
    d = postings_with_city_key(df)
    if d.empty:
        return pd.DataFrame()
    smin = pd.to_numeric(d.get("salary_min_lpa"), errors="coerce")
    smax = pd.to_numeric(d.get("salary_max_lpa"), errors="coerce")
    d = d.assign(_mid=(smin + smax) / 2.0)
    g = d.groupby("city_key", as_index=False).agg(
        postings=("job_title", "count"),
        median_lpa=("_mid", "median"),
        raw_location=("location", lambda s: s.mode().iloc[0] if len(s) else ""),
    )
    ref = load_city_reference()
    if not ref.empty:
        g = g.merge(
            ref[["city_key", "display_name", "lat", "lon", "market_tier_index"]],
            on="city_key",
            how="left",
        )
        g["display_name"] = g["display_name"].fillna(g["raw_location"].astype(str))
        g["market_tier_index"] = g["market_tier_index"].fillna(1).astype(int)
    else:
        g["display_name"] = g["raw_location"].astype(str)
        g["lat"] = float("nan")
        g["lon"] = float("nan")
        g["market_tier_index"] = 1
    return g.sort_values("postings", ascending=False)


def extract_user_skill_phrases(skills_text: str) -> List[str]:
    blob = (skills_text or "").lower()
    return [p for p in skill_phrase_list() if phrase_in_blob(p, blob)]


def skill_match_rate_in_subset(df: pd.DataFrame, phrases: List[str]) -> float:
    """Share of rows where description+title mentions ≥1 user skill phrase."""
    if df.empty or not phrases:
        return 0.0
    hits = 0
    for _, row in df.iterrows():
        blob = row.get("_text", "")
        if any(phrase_in_blob(p, blob) for p in phrases):
            hits += 1
    return hits / len(df)


def national_skill_rates(df: pd.DataFrame, phrases: List[str]) -> Dict[str, float]:
    """Mention rate per phrase across full corpus (for location quotient)."""
    if df.empty or not phrases:
        return {}
    n = len(df)
    out: Dict[str, float] = {}
    for p in phrases:
        c = sum(1 for _, row in df.iterrows() if phrase_in_blob(p, row.get("_text", "")))
        out[p] = c / n if n else 0.0
    return out


def location_quotient(local_rate: float, national_rate: float) -> Optional[float]:
    if national_rate <= 1e-9:
        return None
    return round(local_rate / national_rate, 3)


def skill_location_quotients(
    df: pd.DataFrame, city_key: str, phrases: List[str], top_k: int = 8
) -> pd.DataFrame:
    d = postings_with_city_key(df)
    local = d[d["city_key"] == city_key]
    if local.empty or not phrases:
        return pd.DataFrame()
    nat = national_skill_rates(df, phrases)
    rows = []
    for p in phrases:
        loc_rate = sum(
            1 for _, row in local.iterrows() if phrase_in_blob(p, row.get("_text", ""))
        ) / len(local)
        nrate = nat.get(p, 0.0)
        rows.append(
            {
                "skill": p,
                "local_rate": round(loc_rate, 4),
                "national_rate": round(nrate, 4),
                "lq": location_quotient(loc_rate, nrate),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("lq", ascending=False, na_position="last")
    return out.head(top_k)


def rank_relocation_targets(
    df: pd.DataFrame,
    user_city_key: str,
    phrases: List[str],
    weights: Tuple[float, float] = (0.55, 0.45),
) -> pd.DataFrame:
    """
    Rank cities by a transparent composite: w1 * relative_volume + w2 * skill_match.
    All inputs derived from the posting file.
    """
    agg = aggregate_city_labour_market(df)
    if agg.empty:
        return agg
    base_rows = agg.loc[agg["city_key"] == user_city_key, "postings"]
    base_count = int(base_rows.iloc[0]) if len(base_rows) else max(1, int(agg["postings"].median()))
    d = postings_with_city_key(df)
    w_vol, w_fit = weights
    scores = []
    for _, row in agg.iterrows():
        ck = row["city_key"]
        sub = d[d["city_key"] == ck]
        fit = skill_match_rate_in_subset(sub, phrases)
        vol_ratio = float(row["postings"]) / float(max(1, base_count))
        composite = w_vol * min(vol_ratio, 3.0) / 3.0 + w_fit * fit
        scores.append(
            {
                "city_key": ck,
                "display_name": row.get("display_name", ck),
                "postings": int(row["postings"]),
                "volume_vs_yours": round(row["postings"] / max(1, base_count), 2),
                "your_skill_match_rate": round(fit, 4),
                "score": round(composite, 4),
            }
        )
    out = pd.DataFrame(scores).sort_values("score", ascending=False)
    return out


def relocation_model_delta_pct(
    skills: str,
    education: str,
    experience: int,
    industry: str,
    from_tier_index: int,
    to_tier_index: int,
) -> Tuple[float, float, float]:
    """
    Uses the existing logistic risk model: same profile, only location tier changes.
    Returns (pct_at_from, pct_at_to, delta_pp) where negative delta = lower modeled risk.
    """
    fi = max(0, min(2, int(from_tier_index)))
    ti = max(0, min(2, int(to_tier_index)))
    loc_from = LOCATION_OPTIONS[fi]
    loc_to = LOCATION_OPTIONS[ti]
    r0 = predict_job_risk(skills, education, experience, loc_from, industry)
    r1 = predict_job_risk(skills, education, experience, loc_to, industry)
    d = round(r1.high_risk_probability_pct - r0.high_risk_probability_pct, 1)
    return r0.high_risk_probability_pct, r1.high_risk_probability_pct, d


def build_folium_map(
    agg: pd.DataFrame,
    highlight_city_key: Optional[str] = None,
    extra_marker: Optional[Tuple[float, float, str]] = None,
) -> Any:
    """Create a Folium map centered on India; circle markers sized by posting volume."""
    import folium
    from folium.plugins import HeatMap

    center_lat, center_lon = 22.5, 79.0
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
        attr="© OpenStreetMap contributors, © CARTO",
    )
    if agg.empty:
        return m

    valid = agg.dropna(subset=["lat", "lon"])
    max_p = float(valid["postings"].max()) if len(valid) else 1.0
    heat_data: List[List[float]] = []
    for _, row in valid.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        n = int(row["postings"])
        radius = 6 + 34 * (n / max_p) ** 0.5
        name = str(row.get("display_name", row["city_key"]))
        med = row.get("median_lpa")
        med_s = f"{med:.1f} LPA" if pd.notna(med) else "n/a"
        color = "#6366f1" if row["city_key"] != highlight_city_key else "#06b6d4"
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            weight=2,
            popup=folium.Popup(f"<b>{name}</b><br/>{n} postings<br/>Median: {med_s}", max_width=220),
        ).add_to(m)
        heat_data.append([lat, lon, n])

    if len(heat_data) >= 2:
        HeatMap(heat_data, min_opacity=0.25, radius=18, blur=22, max_zoom=6).add_to(m)

    if extra_marker:
        elat, elon, elabel = extra_marker
        folium.Marker(
            [elat, elon],
            popup=elabel,
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    folium.LatLngPopup().add_to(m)
    return m
