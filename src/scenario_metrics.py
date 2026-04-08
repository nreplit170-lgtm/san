"""
scenario_metrics.py
Computes comparison metrics and scenario quality indices.

Fix: Policy cushion score is now read from PolicyPlaybook (single source of truth).
     The duplicate hardcoded map has been removed.
"""
import pandas as pd
import numpy as np
from src.policy_playbook import PolicyPlaybook


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
        policy_cost_label: str = None,   # kept for backward compatibility, unused
    ) -> dict:
        merged = ScenarioMetrics.compute_delta(baseline_df, scenario_df)

        # Unemployment Stress Index: cumulative excess unemployment-years above baseline
        usi = float(merged["Delta"].clip(lower=0).sum())
        usi = round(usi, 2)

        # Peak deviation from baseline
        peak_delta = round(float(merged["Delta"].max()), 2)

        # Years significantly above baseline (>0.5pp)
        years_above = int((merged["Delta"] > 0.5).sum())

        # Policy cushion — single source of truth from PolicyPlaybook
        policy_cushion = PolicyPlaybook.get_cushion_score(policy_name)

        return {
            "unemployment_stress_index": usi,
            "peak_delta": peak_delta,
            "years_above_baseline": years_above,
            "policy_cushion_score": policy_cushion,
        }

    @staticmethod
    def compute_rqi(scenario_df: pd.DataFrame, recovery_rate: float) -> dict:
        """Recovery Quality Index — characterises speed and sustainability of recovery."""
        if recovery_rate >= 0.45:
            label = "Fast Recovery"
        elif recovery_rate >= 0.30:
            label = "Moderate Recovery"
        elif recovery_rate >= 0.15:
            label = "Slow Recovery"
        else:
            label = "Poor Recovery"

        # Override if trajectory is still rising at end (fragile recovery)
        if len(scenario_df) >= 3:
            last_vals = scenario_df["Scenario_Unemployment"].iloc[-3:].values
            if last_vals[-1] > last_vals[-2] > last_vals[-3]:
                label = "Fast but Fragile"

        return {"rqi_label": label, "recovery_rate_used": recovery_rate}
