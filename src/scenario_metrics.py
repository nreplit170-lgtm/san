"""
scenario_metrics.py
Computes comparison metrics and scenario quality indices.
"""
import pandas as pd
import numpy as np


class ScenarioMetrics:

    @staticmethod
    def compute_delta(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(baseline_df, scenario_df, on="Year", how="inner")
        merged["Delta"] = merged["Scenario_Unemployment"] - merged["Predicted_Unemployment"]
        return merged[["Year", "Predicted_Unemployment", "Scenario_Unemployment", "Delta"]]

    @staticmethod
    def compute_indices(
        baseline_df: pd.DataFrame,
        scenario_df: pd.DataFrame,
        policy_name: str = "None",
        policy_cost_label: str = None,
    ) -> dict:
        merged = ScenarioMetrics.compute_delta(baseline_df, scenario_df)

        # Unemployment Stress Index (USI): cumulative excess unemployment-years
        usi = float(merged["Delta"].clip(lower=0).sum())
        usi = round(usi, 2)

        # Peak delta
        peak_delta = round(float(merged["Delta"].max()), 2)

        # Years significantly above baseline (delta > 0.5pp)
        years_above = int((merged["Delta"] > 0.5).sum())

        # Policy cushion: simple heuristic from policy type
        policy_cushion_map = {
            "Fiscal Stimulus": 35,
            "Monetary Policy": 20,
            "Labor Reforms": 25,
            "Industry Support": 30,
            "None": 0,
        }
        policy_cushion = policy_cushion_map.get(policy_name, 0)

        return {
            "unemployment_stress_index": usi,
            "peak_delta": peak_delta,
            "years_above_baseline": years_above,
            "policy_cushion_score": policy_cushion,
        }

    @staticmethod
    def compute_rqi(scenario_df: pd.DataFrame, recovery_rate: float) -> dict:
        """
        Recovery Quality Index:
        Characterises how fast and sustainably the economy recovers.
        """
        if recovery_rate >= 0.45:
            label = "Fast Recovery"
        elif recovery_rate >= 0.3:
            label = "Moderate Recovery"
        elif recovery_rate >= 0.15:
            label = "Slow Recovery"
        else:
            label = "Poor Recovery"

        # Check if trajectory is stable at the end
        if len(scenario_df) >= 3:
            last_vals = scenario_df["Scenario_Unemployment"].iloc[-3:].values
            if last_vals[-1] > last_vals[-2] > last_vals[-3]:
                label = "Fast but Fragile"

        return {"rqi_label": label, "recovery_rate_used": recovery_rate}
