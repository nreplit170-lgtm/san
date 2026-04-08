"""
data_loader.py

Author: Bhushan Nanavare
Purpose:
    Reliable ingestion of World Bank unemployment data.
    This module is the single source of truth for raw data loading.

Key guarantees:
    - Valid World Bank format
    - Deterministic transformation
    - No silent failures
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    def __init__(self, csv_path: str, country: str):
        self.csv_path = Path(csv_path)
        self.country = country

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found at: {self.csv_path.resolve()}"
            )

    def _load_world_bank_csv(self) -> pd.DataFrame:
        """
        Loads World Bank CSV by skipping metadata rows.
        """
        try:
            df = pd.read_csv(self.csv_path, skiprows=4)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")

        required_columns = {
            "Country Name",
            "Country Code",
            "Indicator Name",
            "Indicator Code",
        }

        if not required_columns.issubset(df.columns):
            raise ValueError(
                "Invalid World Bank CSV format. Required columns missing."
            )

        return df

    def _filter_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset for selected country and
        total unemployment indicator only.
        """
        df_country = df[
            (df["Country Name"] == self.country) &
            (df["Indicator Code"] == "SL.UEM.TOTL.ZS")
     ]

        if df_country.empty:
            raise ValueError(
                f"No unemployment data found for country: {self.country}"
            )

        return df_country


    def _to_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts wide World Bank format to time-series format.
        Output schema:
            Year (int)
            Unemployment_Rate (float)
        """
        df_years = df.drop(
            columns=[
                "Country Name",
                "Country Code",
                "Indicator Name",
                "Indicator Code",
            ]
        )

        df_ts = df_years.melt(
            var_name="Year",
            value_name="Unemployment_Rate"
        )

        df_ts["Year"] = pd.to_numeric(df_ts["Year"], errors="coerce")
        df_ts["Unemployment_Rate"] = pd.to_numeric(
            df_ts["Unemployment_Rate"], errors="coerce"
        )

        df_ts = df_ts.dropna(subset=["Year"])
        df_ts["Year"] = df_ts["Year"].astype(int)

        df_ts = df_ts.sort_values("Year").reset_index(drop=True)

        return df_ts

    def load_clean_data(self) -> pd.DataFrame:
        """
        Full ingestion pipeline.
        """
        raw_df = self._load_world_bank_csv()
        country_df = self._filter_country(raw_df)
        ts_df = self._to_time_series(country_df)

        if ts_df["Unemployment_Rate"].isna().all():
            raise ValueError("All unemployment values are missing.")

        return ts_df
