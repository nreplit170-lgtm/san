"""
shock_scenario.py
Applies an economic shock overlay on top of a baseline forecast.

Economic rationale:
  Real shocks (COVID, financial crises) hit at FULL intensity immediately,
  persist at peak for the shock_duration period, then decay as the economy
  recovers. The old "ramp-up" approach was economically backwards.

  Special case — duration=0:
    A zero-duration shock is an impulse shock: full intensity in year 1 only,
    then decays immediately. Intensity is never silently ignored.
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
        - shock_duration:  number of years the shock persists at FULL peak intensity.
                           0 = impulse shock (hits year 1 at full strength, decays from year 2).
        - recovery_rate:   annual fraction of remaining shock that recovers (0–1).
                           Higher = faster recovery.
        """
        self.shock_intensity = shock_intensity
        self.shock_duration = max(0, shock_duration)
        self.recovery_rate = float(np.clip(recovery_rate, 0.0, 1.0))

    def apply(self, baseline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies shock + recovery overlay to baseline forecast.

        Phase 1 (years 0 … shock_duration-1 inclusive, or just year 0 if duration=0):
            Full shock intensity — unemployment = baseline * (1 + shock_intensity).
        Phase 2 (year shock_duration onward):
            Remaining shock decays by recovery_rate each year.

        Returns a DataFrame with columns: Year, Scenario_Unemployment.
        """
        df = baseline_df.copy()
        n = len(df)

        shock_values = []
        # Remaining shock magnitude — starts at full intensity
        remaining = float(self.shock_intensity)

        # How many years stay at full intensity:
        # duration=0 → peak_years=1 (impulse, one full hit then decay)
        # duration=N → peak_years=N
        peak_years = max(1, self.shock_duration)

        for i in range(n):
            base_val = float(df["Predicted_Unemployment"].iloc[i])

            if i < peak_years:
                # Full shock phase — immediate maximum impact
                current_shock = self.shock_intensity
            else:
                # Recovery phase — shock decays each year
                remaining = remaining * (1.0 - self.recovery_rate)
                remaining = max(remaining, 0.0)
                current_shock = remaining

            scenario_val = base_val * (1.0 + current_shock)
            shock_values.append(round(scenario_val, 4))

        return pd.DataFrame({
            "Year": df["Year"].values,
            "Scenario_Unemployment": shock_values,
        })
