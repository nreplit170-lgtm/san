"""
story_generator.py
Generates a narrative timeline story from scenario vs baseline data.

Fix: Peak year check now runs BEFORE the year-0 stable default, so if the
shock peaks at year 0 it is correctly labelled as a shock event.
"""
import pandas as pd


class StoryGenerator:

    @staticmethod
    def generate_story(scenario_df: pd.DataFrame, baseline_df: pd.DataFrame) -> list:
        """
        Returns a list of story event dicts with keys:
          year, title, body, type, scenario_val, baseline_val, delta
        type is one of: 'shock', 'recovery', 'stable'
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

            # Peak check runs FIRST — no other condition can steal the peak label
            if i == peak_idx and delta > 0.3:
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
                body = f"Unemployment at {scenario_val}%, exceeding baseline by {round(delta, 2)}pp."
            elif delta > 0.1:
                event_type = "recovery"
                title = f"{year}: Gradual Recovery"
                body = f"Unemployment eases to {scenario_val}%, converging toward baseline ({baseline_val}%)."
            elif i == 0:
                # Year-0 stable label only if it genuinely is not a shock
                event_type = "stable"
                title = f"{year}: Economic Baseline"
                body = f"Unemployment stands at {scenario_val}%, closely tracking the baseline of {baseline_val}%."
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
