"""
forecasting.py

Purpose:
    Scenario-aware unemployment forecasting engine.
    Produces real numeric forecasts using historical trends.
    Core model: Trend + Mean-Reversion (replaces unstable pure linear extrapolation).

Economic rationale:
    Pure linear extrapolation assumes trends continue indefinitely, which is
    unrealistic. Unemployment exhibits mean reversion: labor markets adjust,
    policy interventions respond, and extreme rates tend to dissipate. This
    model captures (1) short-term momentum via 10-year trend, and (2) pull
    toward long-run level to prevent runaway forecasts over 6+ years.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Default mean reversion strength: 0 = pure trend, 1 = full reversion to long-run.
# Typical range 0.1–0.3 balances momentum with stability. Documented for tuning.
DEFAULT_MEAN_REVERSION_STRENGTH = 0.15


class ForecastingEngine:
    def __init__(
        self,
        forecast_horizon: int = 5,
        method: str = "exponential_smoothing",
        mean_reversion_strength: float = DEFAULT_MEAN_REVERSION_STRENGTH,
    ):
        """
        Parameters:
        - forecast_horizon: number of future years to predict
        - method: 'linear', 'exponential_smoothing', 'arima_inspired', or 'ensemble'
        - mean_reversion_strength: strength of pull toward long-run unemployment (0–1).
          Higher = forecasts converge faster to historical mean; prevents divergence.
          Default 0.15: moderate reversion. Use 0.2–0.3 for more stable long-term.
        """
        self.forecast_horizon = forecast_horizon
        self.method = method
        self.mean_reversion_strength = mean_reversion_strength

    def _fit_trend(self, df: pd.DataFrame) -> np.poly1d:
        """
        Fits a linear trend using ONLY last 10 years (recent regime).
        Avoids contamination from distant structural shifts.
        """
        window = 10
        recent_df = df.tail(window)

        x = np.arange(len(recent_df))
        y = recent_df["Unemployment_Smoothed"].values

        coeffs = np.polyfit(x, y, deg=1)
        trend_model = np.poly1d(coeffs)

        return trend_model

    def _forecast_trend_reversion(self, df: pd.DataFrame) -> list:
        """
        Recursive multi-step forecast: forecast = trend_prediction + reversion_adjustment.
        Each year is predicted from the previous year's forecast (year-by-year recursion).

        Economic reasoning (why this is more realistic than pure linear extrapolation):
        - Trend: captures short-to-medium momentum from last 10 years.
        - Mean reversion: unemployment tends to gravitate toward a long-run level
          (labor market equilibrium, NAIRU-like). Pure linear extrapolation can
          drift to absurd levels; reversion pulls forecasts back toward history.

        Long-horizon forecasts are less confident because:
        - Structural breaks, policy changes, and shocks accumulate over time.
        - Trend persistence decays: far-out years should depend less on today's
          momentum and more on equilibrium (mean reversion).
        - Uncertainty compounds; recursive forecasts reflect this by increasing
          reversion and reducing trend influence for farther years.
        """
        trend_model = self._fit_trend(df)
        window = 10
        # Slope: annual change from linear trend (robust across numpy versions)
        slope = float(trend_model(1) - trend_model(0))

        # Long-run unemployment level: historical mean over full series.
        long_run_level = float(df["Unemployment_Smoothed"].mean())

        # Max absolute year-over-year change (stability safeguard).
        # Prevents monotonic runaway increases/decreases over many years.
        max_annual_step = 1.5

        # Start from last observed value for recursive chain.
        current_value = float(df["Unemployment_Smoothed"].iloc[-1])
        predictions = []

        for h in range(1, self.forecast_horizon + 1):
            # 1. Trend influence: decays with horizon (farther years = less trend).
            #    Long-horizon forecasts are less confident in trend persistence.
            trend_weight = 1.0 / (1.0 + 0.25 * h)  # e.g. h=1 → 0.8, h=6 → 0.4
            trend_prediction = current_value + slope * trend_weight

            # 2. Reversion influence: increases with horizon.
            #    Far-out forecasts should converge toward long-run level.
            reversion_weight = min(1.0, h / 6.0)
            effective_strength = self.mean_reversion_strength * reversion_weight
            reversion_adjustment = effective_strength * (long_run_level - current_value)

            forecast = trend_prediction + reversion_adjustment
            forecast = max(0.0, forecast)

            # 3. Stability safeguard: prevent monotonic runaway.
            delta = forecast - current_value
            if abs(delta) > max_annual_step:
                forecast = current_value + np.sign(delta) * max_annual_step
                forecast = max(0.0, forecast)

            predictions.append(forecast)
            current_value = forecast  # Recursive: next year starts from this forecast

        return predictions

    def _exponential_smoothing(self, df: pd.DataFrame) -> list:
        """
        Simple exponential smoothing for more responsive forecasts.
        """
        series = df["Unemployment_Smoothed"].values
        alpha = 0.3  # smoothing factor (0.2-0.4 for year-to-year data)
        
        # Calculate smoothed values
        smoothed = [series[0]]
        for i in range(1, len(series)):
            smoothed.append(alpha * series[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast future values
        last_smoothed = smoothed[-1]
        trend = (smoothed[-1] - smoothed[-min(3, len(smoothed))]) / min(3, len(smoothed))
        
        predictions = []
        for i in range(self.forecast_horizon):
            predictions.append(last_smoothed + trend * (i + 1))
        
        return predictions

    def _arima_inspired(self, df: pd.DataFrame) -> list:
        """
        Simple ARIMA-inspired approach: trend + mean reversion.
        """
        series = df["Unemployment_Smoothed"].values
        
        # Trend component (recent change)
        trend = series[-1] - series[-min(5, len(series))]
        
        # Mean reversion component (pull towards historical mean)
        historical_mean = series.mean()
        last_value = series[-1]
        
        predictions = []
        value = last_value
        for i in range(self.forecast_horizon):
            # Mix trend with mean reversion
            reversion_factor = (i + 1) / (self.forecast_horizon + 3)
            value = value + trend * 0.3 - (value - historical_mean) * 0.1 * reversion_factor
            predictions.append(max(0, value))  # Prevent negative unemployment
        
        return predictions

    def _ensemble_forecast(self, df: pd.DataFrame) -> list:
        """
        Combines multiple methods for robust predictions.
        """
        linear_preds = self._forecast_linear(df)
        exp_preds = self._exponential_smoothing(df)
        arima_preds = self._arima_inspired(df)
        
        # Average the three methods
        ensemble = [
            (linear_preds[i] + exp_preds[i] + arima_preds[i]) / 3
            for i in range(self.forecast_horizon)
        ]
        
        return ensemble

    def _forecast_linear(self, df: pd.DataFrame) -> list:
        """
        Trend + mean-reversion forecast (replaces pure linear extrapolation).
        Uses configurable mean_reversion_strength to prevent long-term divergence.
        """
        return self._forecast_trend_reversion(df)

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates future unemployment forecasts using selected method.
        """
        df = df.copy()

        last_year = int(df["Year"].max())

        # Select forecasting method
        if self.method == "linear":
            predictions = self._forecast_linear(df)
        elif self.method == "exponential_smoothing":
            predictions = self._exponential_smoothing(df)
        elif self.method == "arima_inspired":
            predictions = self._arima_inspired(df)
        elif self.method == "ensemble":
            predictions = self._ensemble_forecast(df)
        else:
            # Default to ensemble (most robust)
            predictions = self._ensemble_forecast(df)

        future_years = np.arange(
            last_year + 1,
            last_year + 1 + self.forecast_horizon
        )

        forecast_df = pd.DataFrame({
            "Year": future_years,
            "Predicted_Unemployment": predictions
        })

        return forecast_df
