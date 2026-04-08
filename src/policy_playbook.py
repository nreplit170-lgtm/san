"""
policy_playbook.py
Catalogue of policy interventions and their economic characteristics.
Single source of truth — cushion_score lives HERE, not duplicated elsewhere.
"""

POLICIES = {
    "Fiscal Stimulus": {
        "name": "Fiscal Stimulus",
        "description": "Government spending increases to boost aggregate demand and employment.",
        "relative_cost": "High",
        "effectiveness": "High",
        "time_to_impact": "1–2 years",
        "cushion_score": 35,
        "mechanisms": ["Direct job creation", "Infrastructure investment", "Transfer payments"],
    },
    "Monetary Policy": {
        "name": "Monetary Policy",
        "description": "Central bank lowers interest rates to stimulate borrowing and investment.",
        "relative_cost": "Low",
        "effectiveness": "Moderate",
        "time_to_impact": "6–18 months",
        "cushion_score": 20,
        "mechanisms": ["Rate cuts", "Quantitative easing", "Credit easing"],
    },
    "Labor Reforms": {
        "name": "Labor Reforms",
        "description": "Structural reforms to improve labor market flexibility and skills matching.",
        "relative_cost": "Moderate",
        "effectiveness": "Moderate–High (long run)",
        "time_to_impact": "2–5 years",
        "cushion_score": 25,
        "mechanisms": ["Re-skilling programs", "Hiring incentives", "Unemployment insurance reform"],
    },
    "Industry Support": {
        "name": "Industry Support",
        "description": "Targeted subsidies and tax relief to hard-hit industries to prevent mass layoffs.",
        "relative_cost": "Moderate",
        "effectiveness": "Moderate",
        "time_to_impact": "Immediate",
        "cushion_score": 30,
        "mechanisms": ["Sectoral grants", "Wage subsidies", "Loan guarantees"],
    },
    "None": {
        "name": "No Policy",
        "description": "No specific policy intervention; market self-correction.",
        "relative_cost": "None",
        "effectiveness": "Low",
        "time_to_impact": "N/A",
        "cushion_score": 0,
        "mechanisms": [],
    },
}


class PolicyPlaybook:
    @staticmethod
    def get_policy(policy_name: str) -> dict:
        if not policy_name or policy_name == "None":
            return POLICIES["None"]
        return POLICIES.get(policy_name, POLICIES["None"])

    @staticmethod
    def get_cushion_score(policy_name: str) -> int:
        return PolicyPlaybook.get_policy(policy_name).get("cushion_score", 0)

    @staticmethod
    def list_policies() -> list:
        return list(POLICIES.keys())
