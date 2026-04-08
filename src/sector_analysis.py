"""
sector_analysis.py
Relative Sector Stress Index (RSSI) and resilience scoring.
"""
import pandas as pd
import numpy as np


SECTOR_CONFIG = {
    "Healthcare": {"shock_sensitivity": 0.4, "base_resilience": 75},
    "IT": {"shock_sensitivity": 0.55, "base_resilience": 70},
    "Services": {"shock_sensitivity": 0.75, "base_resilience": 45},
    "Manufacturing": {"shock_sensitivity": 0.85, "base_resilience": 40},
    "Construction": {"shock_sensitivity": 0.90, "base_resilience": 35},
}


class SectorAnalysis:

    @staticmethod
    def analyze_sectors(
        scenario_df: pd.DataFrame,
        shock_intensity: float,
        recovery_rate: float,
    ) -> pd.DataFrame:
        """
        Computes Stress_Score and Resilience_Score for each sector.
        """
        records = []
        peak_unemployment = float(scenario_df["Scenario_Unemployment"].max())

        for sector, cfg in SECTOR_CONFIG.items():
            sensitivity = cfg["shock_sensitivity"]
            base_resilience = cfg["base_resilience"]

            # Stress Score: scaled by shock and sector sensitivity
            stress = min(100, round(peak_unemployment * sensitivity * (1 + shock_intensity) * 5, 1))

            # Resilience Score: reduced by shock, boosted by recovery
            resilience = round(max(0, base_resilience - stress * 0.3 + recovery_rate * 20), 1)
            resilience = min(100, resilience)

            records.append({
                "Sector": sector,
                "Stress_Score": stress,
                "Resilience_Score": resilience,
                "Sensitivity": round(sensitivity, 2),
            })

        return pd.DataFrame(records)
