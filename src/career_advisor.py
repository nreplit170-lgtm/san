"""
career_advisor.py
Rule-based AI Career & Skill Advisor.

Fix: Thresholds for "growth" vs "risk" classification now scale dynamically
with shock_intensity so that a severe crisis produces appropriately pessimistic
advice rather than falsely optimistic sector labels.
"""
import pandas as pd


class CareerAdvisor:

    SECTOR_SKILLS = {
        "Healthcare":    ["Telemedicine", "Geriatric Care", "Biotech", "Health Informatics"],
        "IT":            ["AI/ML", "Cybersecurity", "Cloud Computing", "Data Engineering"],
        "Services":      ["Digital Marketing", "E-commerce Management", "CX Automation", "Crisis Management"],
        "Manufacturing": ["Robotics", "Supply Chain Analytics", "Lean Six Sigma", "IoT Maintenance"],
        "Construction":  ["Green Building", "Project Management", "BIM (Building Info Modeling)", "Sustainable Urban Planning"],
    }

    @staticmethod
    def _dynamic_thresholds(shock_intensity: float) -> dict:
        """
        As shock intensifies, the bar for being a "growth" sector rises
        and the bar for being "at risk" falls — matching economic reality.

        shock_intensity:  0.0 → 0.2   low shock   → generous thresholds
                          0.2 → 0.4   moderate    → tightened thresholds
                          0.4+        severe/crisis → strict thresholds
        """
        if shock_intensity <= 0.2:
            return {"growth_resilience": 60, "growth_stress": 50, "risk_stress": 65}
        elif shock_intensity <= 0.4:
            return {"growth_resilience": 68, "growth_stress": 40, "risk_stress": 55}
        else:
            return {"growth_resilience": 75, "growth_stress": 30, "risk_stress": 45}

    @staticmethod
    def generate_advice(sector_impact_df: pd.DataFrame, shock_intensity: float = 0.0) -> dict:
        """
        Generates career advice based on sector stress and resilience.
        shock_intensity is used to set dynamic classification thresholds.
        """
        thresholds = CareerAdvisor._dynamic_thresholds(shock_intensity)
        g_res = thresholds["growth_resilience"]
        g_str = thresholds["growth_stress"]
        r_str = thresholds["risk_stress"]

        sectors = sector_impact_df.to_dict(orient="records")

        advice = {
            "growth_sectors": [],
            "risk_sectors": [],
            "recommended_skills": [],
            "upskilling_pathways": [],
            "shock_severity": (
                "Low" if shock_intensity <= 0.2 else
                "Moderate" if shock_intensity <= 0.4 else
                "Severe"
            ),
        }

        for s in sectors:
            name = s["Sector"]
            resilience = s["Resilience_Score"]
            stress = s["Stress_Score"]

            if resilience > g_res and stress < g_str:
                advice["growth_sectors"].append(name)
                skills = CareerAdvisor.SECTOR_SKILLS.get(name, [])
                advice["recommended_skills"].extend(skills[:3])
            elif stress > r_str:
                advice["risk_sectors"].append(name)

        advice["recommended_skills"] = list(set(advice["recommended_skills"]))

        narrative = []
        severity = advice["shock_severity"]
        if advice["growth_sectors"]:
            narrative.append(
                f"Under {severity.lower()} shock conditions, strongest growth potential remains in: "
                f"{', '.join(advice['growth_sectors'])}."
            )
        else:
            narrative.append(
                f"Under {severity.lower()} economic shock, no sector currently meets the growth threshold. "
                "Focus on resilience-building and transferable skills."
            )

        if advice["risk_sectors"]:
            narrative.append(
                f"High stress observed in: {', '.join(advice['risk_sectors'])}. "
                "Professionals in these areas should prioritize upskilling and pivot planning."
            )

        advice["narrative"] = " ".join(narrative)
        return advice
