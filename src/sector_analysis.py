"""
sector_analysis.py
Relative Sector Stress Index (RSSI) and resilience scoring.

Fix: Stress is now normalized against India's own historical peak unemployment,
not an arbitrary ×5 multiplier. This ensures scores are meaningful relative
to actual Indian labor market history.
"""
import pandas as pd
import numpy as np

SECTOR_CONFIG = {
    "Healthcare":    {"shock_sensitivity": 0.40, "base_resilience": 75},
    "IT":            {"shock_sensitivity": 0.55, "base_resilience": 70},
    "Services":      {"shock_sensitivity": 0.75, "base_resilience": 45},
    "Manufacturing": {"shock_sensitivity": 0.85, "base_resilience": 40},
    "Construction":  {"shock_sensitivity": 0.90, "base_resilience": 35},
}

# India's approximate historical worst unemployment rate (World Bank).
# Used as the normalization anchor so that "100% stress" = historically worst level.
INDIA_HISTORICAL_PEAK_UNEMPLOYMENT = 9.5


class SectorAnalysis:

    @staticmethod
    def analyze_sectors(
        scenario_df: pd.DataFrame,
        shock_intensity: float,
        recovery_rate: float,
        historical_peak: float = INDIA_HISTORICAL_PEAK_UNEMPLOYMENT,
    ) -> pd.DataFrame:
        """
        Computes Stress_Score (0–100) and Resilience_Score (0–100) for each sector.

        Stress is normalized: 100 = sector stress equivalent to the worst historical
        Indian unemployment, calibrated by sector sensitivity.
        """
        records = []
        peak_unemployment = float(scenario_df["Scenario_Unemployment"].max())

        # Normalize: how severe is the scenario peak relative to India's worst?
        severity_ratio = np.clip(peak_unemployment / historical_peak, 0.0, 1.5)

        for sector, cfg in SECTOR_CONFIG.items():
            sensitivity = cfg["shock_sensitivity"]
            base_resilience = cfg["base_resilience"]

            # Stress = severity × sensitivity × shock amplifier, capped at 100
            raw_stress = severity_ratio * sensitivity * (1.0 + shock_intensity * 0.5) * 100.0
            stress = round(min(100.0, raw_stress), 1)

            # Resilience reduced by stress proportion, boosted by fast recovery
            resilience = round(max(0.0, base_resilience - stress * 0.35 + recovery_rate * 25.0), 1)
            resilience = min(100.0, resilience)

            records.append({
                "Sector": sector,
                "Stress_Score": stress,
                "Resilience_Score": resilience,
                "Sensitivity": round(sensitivity, 2),
            })

        return pd.DataFrame(records)
