
import pandas as pd

class CareerAdvisor:
    """
    Rule-based AI Career & Skill Advisor.
    Maps sector conditions to skill recommendations.
    """
    
    SECTOR_SKILLS = {
        "Healthcare": ["Telemedicine", "Geriatric Care", "Biotech", "Health Informatics"],
        "IT": ["AI/ML", "Cybersecurity", "Cloud Computing", "Data Engineering"],
        "Services": ["Digital Marketing", "E-commerce Management", "CX Automation", "Crisis Management"],
        "Manufacturing": ["Robotics", "Supply Chain Analytics", "Lean Six Sigma", "IoT Maintenance"],
        "Construction": ["Green Building", "Project Management", "BIM (Building Info Modeling)", "Sustainable Urban Planning"]
    }
    
    @staticmethod
    def generate_advice(sector_impact_df: pd.DataFrame) -> dict:
        """
        Generates career advice based on sector stress and resilience.
        """
        # Convert df to dict for easier processing
        sectors = sector_impact_df.to_dict(orient="records")
        
        advice = {
            "growth_sectors": [],
            "risk_sectors": [],
            "recommended_skills": [],
            "upskilling_pathways": []
        }
        
        for s in sectors:
            name = s["Sector"]
            resilience = s["Resilience_Score"]
            stress = s["Stress_Score"]
            
            # Logic for categorization
            # Growth: High Resilience (>60) AND Moderate-to-Low Stress (<50)
            if resilience > 60 and stress < 50:
                advice["growth_sectors"].append(name)
                # Add top 2 skills for brevity
                skills = CareerAdvisor.SECTOR_SKILLS.get(name, [])
                advice["recommended_skills"].extend(skills[:3])
                
            elif stress > 65:
                advice["risk_sectors"].append(name)
            
        # Deduplicate skills
        advice["recommended_skills"] = list(set(advice["recommended_skills"]))
        
        # Generate narrative text
        narrative = []
        if advice["growth_sectors"]:
            narrative.append(f"Strongest growth potential seen in: {', '.join(advice['growth_sectors'])}.")
        else:
            narrative.append("Market conditions are tough across most sectors.")

        if advice["risk_sectors"]:
            narrative.append(f"High stress observed in: {', '.join(advice['risk_sectors'])}. Professionals in these areas should prioritize resilience-building skills.")
            
        advice["narrative"] = " ".join(narrative)
        
        return advice
