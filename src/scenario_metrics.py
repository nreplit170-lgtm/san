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
        """Recovery Quality Index — characterises speed and sustainability of recovery.

        Label is now derived from the actual scenario OUTPUT trajectory, not the input
        recovery_rate slider. Two scenarios with the same recovery_rate but different
        shock intensities previously received identical labels despite very different
        real outcomes. The recovered_fraction metric measures what share of the
        peak shock excess was unwound by the end of the forecast horizon.
        """
        vals = scenario_df["Scenario_Unemployment"].values
        peak = float(vals.max())
        end = float(vals[-1])
        start = float(vals[0])

        # What fraction of the peak-above-start excess was recovered by end of period?
        shock_size = peak - start
        if shock_size > 1e-6:
            recovered_fraction = float(np.clip((peak - end) / shock_size, 0.0, 1.0))
        else:
            recovered_fraction = 1.0   # negligible shock — trivially "recovered"

        if recovered_fraction >= 0.70:
            label = "Fast Recovery"
        elif recovered_fraction >= 0.40:
            label = "Moderate Recovery"
        elif recovered_fraction >= 0.20:
            label = "Slow Recovery"
        else:
            label = "Poor Recovery"

        # Fragile override: trajectory still rising monotonically at end of horizon.
        if len(vals) >= 3:
            if vals[-1] > vals[-2] > vals[-3]:
                label = "Fast but Fragile"

        return {
            "rqi_label": label,
            "recovery_rate_used": recovery_rate,
            "recovered_fraction": round(recovered_fraction, 3),
        }
