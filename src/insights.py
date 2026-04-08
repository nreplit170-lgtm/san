"""
insights.py

Purpose:
    Convert forecasts and risk metrics into
    interpretable insights and explanations.
"""

import pandas as pd


class InsightsEngine:
    def summarize_regimes(self, df: pd.DataFrame) -> dict:
        """
        Summarizes economic regimes in historical data.
        """
        regime_counts = df["Regime"].value_counts().to_dict()

        dominant_regime = max(regime_counts, key=regime_counts.get)

        return {
            "regime_distribution": regime_counts,
            "dominant_regime": dominant_regime
        }

    def summarize_risk(self, forecast_df: pd.DataFrame) -> dict:
        """
        Summarizes future unemployment risk.
        """
        risk_counts = forecast_df["Risk_Level"].value_counts().to_dict()

        dominant_risk = max(risk_counts, key=risk_counts.get)

        avg_uncertainty = forecast_df["Uncertainty"].mean()

        return {
            "risk_distribution": risk_counts,
            "dominant_risk": dominant_risk,
            "average_uncertainty": round(avg_uncertainty, 3)
        }

    def generate_insights(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame
    ) -> list:
        """
        Generates human-readable insights.
        """
        insights = []

        # Regime insight
        regime_summary = self.summarize_regimes(historical_df)
        insights.append(
            f"Historical data is dominated by '{regime_summary['dominant_regime']}' periods, "
            f"with notable shock events affecting unemployment trends."
        )

        # Risk insight
        risk_summary = self.summarize_risk(forecast_df)
        insights.append(
            f"Future unemployment risk is assessed as '{risk_summary['dominant_risk']}', "
            f"driven by historical volatility and observed shock events."
        )

        # Uncertainty insight
        insights.append(
            f"The average forecast uncertainty is approximately "
            f"{risk_summary['average_uncertainty']} percentage points, "
            f"indicating limited predictability under current conditions."
        )

        # Policy-style insight
        if risk_summary["dominant_risk"] == "High":
            insights.append(
                "Policy interventions and employment stabilization measures "
                "may be required to mitigate future unemployment risk."
            )
        else:
            insights.append(
                "Current trends suggest manageable unemployment dynamics, "
                "though continued monitoring is recommended."
            )

        return insights
