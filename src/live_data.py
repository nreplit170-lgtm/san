"""
live_data.py
Fetches current unemployment and sector data from the World Bank Open API
(free, no API key required).
Falls back to local CSV for unemployment when the API is unavailable.

World Bank API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

WB_API = "https://api.worldbank.org/v2/country/{iso}/indicator/SL.UEM.TOTL.ZS"
WB_INDICATOR_API = "https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}"
FALLBACK_CSV = Path("data/raw/india_unemployment.csv")
COUNTRY_ISO = {"India": "IN"}

# ── World Bank sector indicators ───────────────────────────────────────────────
# These three sector categories (Agriculture, Industry, Services) are the standard
# World Bank / ILO breakdown available for India.
# Manufacturing is a sub-component of Industry and has its own dedicated series.
SECTOR_INDICATORS = {
    "Agriculture": {
        "employment": "SL.AGR.EMPL.ZS",   # % of total employment
        "gdp":        "NV.AGR.TOTL.ZS",   # % of GDP
    },
    "Industry": {
        "employment": "SL.IND.EMPL.ZS",
        "gdp":        "NV.IND.TOTL.ZS",
    },
    "Manufacturing": {
        "employment": None,                # ILO sub-series not available via WB
        "gdp":        "NV.IND.MANF.ZS",   # % of GDP
    },
    "Services": {
        "employment": "SL.SRV.EMPL.ZS",
        "gdp":        "NV.SRV.TOTL.ZS",
    },
    "Construction": {
        "employment": None,
        "gdp":        "NV.CON.TOTL.ZS",   # % of GDP (not always available)
    },
}

# World Bank unemployment data is updated annually.
# Cache entries expire after 24 hours so a long-running server
# eventually refreshes rather than serving indefinitely stale data.
_CACHE_TTL_SECONDS = 86_400   # 24 hours

_cache: dict = {}          # key → {"df": pd.DataFrame, "ts": float}


def fetch_world_bank(country: str = "India", per_page: int = 65) -> pd.DataFrame:
    """
    Pulls unemployment rate (SL.UEM.TOTL.ZS) from World Bank API.
    Returns DataFrame with columns: Year (int), Unemployment_Rate (float).
    Falls back to local CSV on any network error.
    """
    iso = COUNTRY_ISO.get(country, "IN")
    cache_key = f"{country}_{per_page}"

    entry = _cache.get(cache_key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL_SECONDS:
        return entry["df"]

    url = WB_API.format(iso=iso)
    params = {"format": "json", "per_page": per_page, "mrv": per_page}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data or len(data) < 2 or not data[1]:
            raise ValueError("Empty response from World Bank API")

        records = []
        for entry_raw in data[1]:
            yr = entry_raw.get("date")
            val = entry_raw.get("value")
            if yr and val is not None:
                try:
                    records.append({"Year": int(yr), "Unemployment_Rate": float(val)})
                except (ValueError, TypeError):
                    continue

        if not records:
            raise ValueError("No valid records in API response")

        df = pd.DataFrame(records).sort_values("Year").reset_index(drop=True)
        df = df.dropna(subset=["Unemployment_Rate"])
        df = df[df["Unemployment_Rate"] > 0]

        _cache[cache_key] = {"df": df, "ts": time.time()}
        return df

    except Exception:
        return _load_fallback(country)


def _load_fallback(country: str = "India") -> pd.DataFrame:
    """Load local CSV as fallback when API is unavailable."""
    try:
        from src.data_loader import DataLoader
        df = DataLoader(str(FALLBACK_CSV), country).load_clean_data()
        return df
    except Exception:
        return pd.DataFrame(columns=["Year", "Unemployment_Rate"])


def get_data_source_label(country: str = "India") -> str:
    """Returns whether data came from live API or local fallback.

    Checks the in-process cache first — if fetch_world_bank() already succeeded
    this session the answer is known without making a second HTTP call.
    """
    # If ANY per_page key for this country is in a valid (non-expired) cache entry,
    # the API was reachable and we know the answer without a second HTTP call.
    now = time.time()
    for key, val in _cache.items():
        if key.startswith(f"{country}_") and (now - val["ts"]) < _CACHE_TTL_SECONDS:
            return "🟢 Live — World Bank API"
    # Cache miss — make a minimal probe request.
    iso = COUNTRY_ISO.get(country, "IN")
    try:
        resp = requests.get(
            WB_API.format(iso=iso),
            params={"format": "json", "per_page": 1, "mrv": 1},
            timeout=5,
        )
        if resp.status_code == 200:
            return "🟢 Live — World Bank API"
    except Exception:
        pass
    return "🟡 Offline — Local CSV"


def clear_cache() -> None:
    """Evict all cached entries (useful in tests or when forcing a refresh)."""
    _cache.clear()


# ── Live sector data ───────────────────────────────────────────────────────────

def _fetch_single_indicator(
    indicator: str,
    iso: str = "IN",
    mrv: int = 10,
) -> Optional[float]:
    """
    Pull the most-recent non-null value for a single World Bank indicator.
    Returns the value as a float, or None if unavailable.
    We ask for up to `mrv` most-recent values and take the first non-null one,
    because some series have a 1-2 year reporting lag.
    """
    url = WB_INDICATOR_API.format(iso=iso, indicator=indicator)
    params = {"format": "json", "mrv": mrv}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data or len(data) < 2 or not data[1]:
            return None
        for entry in data[1]:
            val = entry.get("value")
            if val is not None:
                return float(val)
    except Exception:
        pass
    return None


def fetch_sector_indicators(country: str = "India") -> pd.DataFrame:
    """
    Fetch the latest World Bank sector employment-share and GDP-share figures
    for India and return a tidy DataFrame.

    Columns:
        Sector              — sector name
        Employment_Share    — % of total employment (float or NaN)
        GDP_Share           — % of GDP (float or NaN)
        Source              — "World Bank (live)" or "Unavailable"

    Falls back gracefully: if an indicator is unavailable its column is NaN.
    The entire function never raises — worst case it returns an empty DataFrame.
    """
    cache_key = f"sector_indicators_{country}"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL_SECONDS:
        return entry["df"]

    iso = COUNTRY_ISO.get(country, "IN")
    rows = []
    any_live = False

    for sector, codes in SECTOR_INDICATORS.items():
        emp_code = codes.get("employment")
        gdp_code = codes.get("gdp")

        emp_val = _fetch_single_indicator(emp_code, iso) if emp_code else None
        gdp_val = _fetch_single_indicator(gdp_code, iso) if gdp_code else None

        if emp_val is not None or gdp_val is not None:
            any_live = True

        rows.append({
            "Sector":           sector,
            "Employment_Share": round(emp_val, 2) if emp_val is not None else float("nan"),
            "GDP_Share":        round(gdp_val, 2) if gdp_val is not None else float("nan"),
            "Source":           "World Bank (live)" if (emp_val or gdp_val) else "Unavailable",
        })

    if not rows:
        return pd.DataFrame(columns=["Sector", "Employment_Share", "GDP_Share", "Source"])

    df = pd.DataFrame(rows)
    if any_live:
        _cache[cache_key] = {"df": df, "ts": time.time()}
    return df
