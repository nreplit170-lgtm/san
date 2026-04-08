"""
story_generator.py
Generates a narrative timeline story from scenario vs baseline data.
"""
import pandas as pd


class StoryGenerator:

    @staticmethod
    def generate_story(scenario_df: pd.DataFrame, baseline_df: pd.DataFrame) -> list:
        """
        Produces a list of story events describing the economic trajectory.
        Each event is a dict with: year, title, body, type (shock/recovery/stable).
        """
        merged = pd.merge(baseline_df, scenario_df, on="Year", how="inner")
        merged["Delta"] = merged["Scenario_Unemployment"] - merged["Predicted_Unemployment"]

        story = []
        peak_idx = merged["Scenario_Unemployment"].idxmax()

        for i, row in merged.iterrows():
            year = int(row["Year"])
            delta = row["Delta"]
            scenario_val = round(row["Scenario_Unemployment"], 2)
            baseline_val = round(row["Predicted_Unemployment"], 2)

            if i == 0 and delta < 0.1:
                event_type = "stable"
                title = f"{year}: Economic Baseline"
                body = f"Unemployment stands at {scenario_val}%, closely tracking the baseline of {baseline_val}%."
            elif i == peak_idx and delta > 0.3:
                event_type = "shock"
                title = f"{year}: Peak Unemployment Shock"
                body = (
                    f"Unemployment peaks at {scenario_val}% — "
                    f"{round(delta, 2)}pp above baseline ({baseline_val}%). "
                    "Labor markets face maximum strain."
                )
            elif delta > 0.5:
                event_type = "shock"
                title = f"{year}: Elevated Stress"
                body = f"Unemployment at {scenario_val}%, exceeding baseline by {round(delta,2)}pp."
            elif delta > 0.1:
                event_type = "recovery"
                title = f"{year}: Gradual Recovery"
                body = f"Unemployment eases to {scenario_val}%, converging toward baseline ({baseline_val}%)."
            else:
                event_type = "stable"
                title = f"{year}: Stabilisation"
                body = f"Unemployment at {scenario_val}%, near baseline levels. Recovery appears sustained."

            story.append({
                "year": year,
                "title": title,
                "body": body,
                "type": event_type,
                "scenario_val": scenario_val,
                "baseline_val": baseline_val,
                "delta": round(delta, 2),
            })

        return story
