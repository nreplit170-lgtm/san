"""
live_data.py
Fetches current unemployment data from the World Bank Open API (free, no key required).
Falls back to the local CSV if the API is unavailable.

World Bank API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""
import requests
import pandas as pd
from pathlib import Path

WB_API = "https://api.worldbank.org/v2/country/{iso}/indicator/SL.UEM.TOTL.ZS"
FALLBACK_CSV = Path("data/raw/india_unemployment.csv")
COUNTRY_ISO = {"India": "IN"}

_cache: dict = {}


def fetch_world_bank(country: str = "India", per_page: int = 65) -> pd.DataFrame:
    """
    Pulls unemployment rate (SL.UEM.TOTL.ZS) from World Bank API.
    Returns DataFrame with columns: Year (int), Unemployment_Rate (float).
    Falls back to local CSV on any network error.
    """
    iso = COUNTRY_ISO.get(country, "IN")
    cache_key = f"{country}_{per_page}"

    if cache_key in _cache:
        return _cache[cache_key]

    url = WB_API.format(iso=iso)
    params = {"format": "json", "per_page": per_page, "mrv": per_page}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data or len(data) < 2 or not data[1]:
            raise ValueError("Empty response from World Bank API")

        records = []
        for entry in data[1]:
            yr = entry.get("date")
            val = entry.get("value")
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

        _cache[cache_key] = df
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
    """Returns whether data came from live API or local fallback."""
    iso = COUNTRY_ISO.get(country, "IN")
    try:
        resp = requests.get(
            WB_API.format(iso=iso),
            params={"format": "json", "per_page": 1, "mrv": 1},
            timeout=5
        )
        if resp.status_code == 200:
            return "🟢 Live — World Bank API"
    except Exception:
        pass
    return "🟡 Offline — Local CSV"


def clear_cache():
    _cache.clear()
