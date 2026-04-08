
import pandas as pd

class InsightGenerator:
    """
    Rule-based AI Insight Engine.
    Interprets scenario results into human-readable text.
    No LLMs, just logic.
    """

    @staticmethod
    def generate_scenario_insights(
        scenario_name: str,
        indices: dict,
        sector_impact: pd.DataFrame
    ) -> dict:
        """
        Generates structured text insights for a specific scenario.
        """
        
        # 1. Macro Insight
        usi = indices.get("unemployment_stress_index", 0)
        peak_delta = indices.get("peak_delta", 0)
        peak_year_idx = indices.get("years_above_baseline", 0) # This is actually duration, not year index.
        # We don't have the exact year of peak in indices, but we can infer or it's fine to be general.
        
        if usi > 30:
            macro_tone = "severe economic stress"
        elif usi > 15:
            macro_tone = "moderate economic headwinds"
        else:
            macro_tone = "manageable market fluctuations"
            
        macro_text = (
            f"Under the {scenario_name} conditions, the labor market faces {macro_tone}. "
            f"Unemployment is projected to deviate from the baseline by up to {peak_delta} percentage points."
        )

        # 2. Sector Insight
        # Identify most and least stressed sectors
        sorted_sectors = sector_impact.sort_values(by="Stress_Score", ascending=False)
        most_stressed = sorted_sectors.iloc[0]
        least_stressed = sorted_sectors.iloc[-1]
        
        sector_text = (
            f"{most_stressed['Sector']} experiences the highest stress (Score: {most_stressed['Stress_Score']}), "
            f"driven by its high sensitivity to economic shocks. "
            f"Conversely, {least_stressed['Sector']} remains relatively resilient."
        )

        # 3. Recovery Insight
        rqi_label = indices.get("rqi_label", "Unknown")
        recovery_text = f"The recovery trajectory is classified as '{rqi_label}', indicating the speed and stability of returning to pre-shock levels."

        return {
            "summary": f"{macro_text} {sector_text} {recovery_text}",
            "macro_insight": macro_text,
            "sector_insight": sector_text,
            "recovery_insight": recovery_text
        }
