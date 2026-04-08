"""
shock_scenario.py
Applies an economic shock overlay on top of a baseline forecast.
"""
import pandas as pd
import numpy as np


class ShockScenario:
    def __init__(
        self,
        shock_intensity: float = 0.3,
        shock_duration: int = 2,
        recovery_rate: float = 0.3,
    ):
        """
        Parameters:
        - shock_intensity: fractional increase in unemployment (0.3 = +30% of baseline)
        - shock_duration:  number of years the shock persists at peak
        - recovery_rate:   annual fraction of shock that recovers (0–1)
        """
        self.shock_intensity = shock_intensity
        self.shock_duration = shock_duration
        self.recovery_rate = recovery_rate

    def apply(self, baseline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies shock + recovery overlay to baseline forecast.

        Returns a new DataFrame with Year and Scenario_Unemployment columns.
        """
        df = baseline_df.copy()
        n = len(df)

        shock_values = []
        current_shock = 0.0

        for i in range(n):
            base_val = df["Predicted_Unemployment"].iloc[i]

            if i < self.shock_duration:
                # Build up to peak shock
                ramp = (i + 1) / max(self.shock_duration, 1)
                current_shock = self.shock_intensity * ramp
            else:
                # Decay the shock
                current_shock = current_shock * (1.0 - self.recovery_rate)
                current_shock = max(current_shock, 0.0)

            scenario_val = base_val * (1.0 + current_shock)
            shock_values.append(round(scenario_val, 4))

        result = pd.DataFrame({
            "Year": df["Year"].values,
            "Scenario_Unemployment": shock_values,
        })

        return result
