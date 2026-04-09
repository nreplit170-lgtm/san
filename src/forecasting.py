"""
forecasting.py

Purpose:
    Scenario-aware unemployment forecasting engine.
    Default method: 'ensemble' — averages linear trend+mean-reversion,
    exponential smoothing, and ARIMA-inspired for the most robust result.

Economic rationale:
    Pure linear extrapolation assumes trends continue indefinitely, which is
    unrealistic. Unemployment exhibits mean reversion: labor markets adjust,
    policy interventions respond, and extreme rates tend to dissipate. This
    model captures (1) short-term momentum via 10-year trend, and (2) pull
    toward long-run level to prevent runaway forecasts over 6+ years.
"""

import pandas as pd
import numpy as np

DEFAULT_MEAN_REVERSION_STRENGTH = 0.15


class ForecastingEngine:
    def __init__(
        self,
        forecast_horizon: int = 5,
        method: str = "ensemble",
        mean_reversion_strength: float = DEFAULT_MEAN_REVERSION_STRENGTH,
    ):
        """
        Parameters:
        - forecast_horizon: number of future years to predict
        - method: 'linear', 'exponential_smoothing', 'arima_inspired', or 'ensemble'
                  Default is 'ensemble' — most robust, uses mean reversion.
        - mean_reversion_strength: strength of pull toward long-run unemployment (0–1).
        """
        self.forecast_horizon = forecast_horizon
        self.method = method
        self.mean_reversion_strength = mean_reversion_strength

    def _fit_trend(self, df: pd.DataFrame) -> np.poly1d:
        window = 10
        recent_df = df.tail(window)
        x = np.arange(len(recent_df))
        y = recent_df["Unemployment_Smoothed"].values
        coeffs = np.polyfit(x, y, deg=1)
        return np.poly1d(coeffs)

    def _forecast_trend_reversion(self, df: pd.DataFrame) -> list:
        """
        Trend + Mean-Reversion forecast (the core well-designed model).
        Each year: prediction = trend step + reversion pull toward long-run mean.
        """
        trend_model = self._fit_trend(df)
        slope = float(trend_model(1) - trend_model(0))
        long_run_level = float(df["Unemployment_Smoothed"].mean())
        max_annual_step = 1.5
        current_value = float(df["Unemployment_Smoothed"].iloc[-1])
        predictions = []

        for h in range(1, self.forecast_horizon + 1):
            trend_weight = 1.0 / (1.0 + 0.25 * h)
            trend_prediction = current_value + slope * trend_weight
            reversion_weight = min(1.0, h / 6.0)
            effective_strength = self.mean_reversion_strength * reversion_weight
            reversion_adjustment = effective_strength * (long_run_level - current_value)
            forecast = trend_prediction + reversion_adjustment
            forecast = max(0.0, forecast)
            delta = forecast - current_value
            if abs(delta) > max_annual_step:
                forecast = current_value + np.sign(delta) * max_annual_step
                forecast = max(0.0, forecast)
            predictions.append(forecast)
            current_value = forecast

        return predictions

    def _exponential_smoothing(self, df: pd.DataFrame) -> list:
        series = df["Unemployment_Smoothed"].values
        alpha = 0.3
        smoothed = [series[0]]
        for i in range(1, len(series)):
            smoothed.append(alpha * series[i] + (1 - alpha) * smoothed[i - 1])
        last_smoothed = smoothed[-1]
        trend = (smoothed[-1] - smoothed[-min(3, len(smoothed))]) / min(3, len(smoothed))
        return [max(0.0, last_smoothed + trend * (i + 1)) for i in range(self.forecast_horizon)]

    def _arima_inspired(self, df: pd.DataFrame) -> list:
        series = df["Unemployment_Smoothed"].values
        # Compute per-step (annual) trend, not the raw N-year total difference.
        # The old code used the 5-year total and multiplied by 0.3 each step,
        # which added ~1.2× the correct annual rate each year instead of ~0.3×.
        n_lookback = min(5, len(series))
        n_intervals = max(1, n_lookback - 1)
        annual_trend = (series[-1] - series[-n_lookback]) / n_intervals
        historical_mean = series.mean()
        last_value = series[-1]
        predictions = []
        value = last_value
        for i in range(self.forecast_horizon):
            reversion_factor = (i + 1) / (self.forecast_horizon + 3)
            # 0.3 is the AR dampening weight on the annual trend component.
            value = value + annual_trend * 0.3 - (value - historical_mean) * 0.1 * reversion_factor
            predictions.append(max(0.0, value))
        return predictions

    def _ensemble_forecast(self, df: pd.DataFrame) -> list:
        """
        Weighted ensemble: trend+reversion (50%), ARIMA-inspired (30%), exp smoothing (20%).
        Favours the mean-reversion model which has the strongest economic grounding.
        """
        tr_preds = self._forecast_trend_reversion(df)
        exp_preds = self._exponential_smoothing(df)
        arima_preds = self._arima_inspired(df)
        return [
            0.50 * tr_preds[i] + 0.30 * arima_preds[i] + 0.20 * exp_preds[i]
            for i in range(self.forecast_horizon)
        ]

    def _forecast_linear(self, df: pd.DataFrame) -> list:
        return self._forecast_trend_reversion(df)

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        last_year = int(df["Year"].max())

        method_map = {
            "linear": self._forecast_trend_reversion,
            "exponential_smoothing": self._exponential_smoothing,
            "arima_inspired": self._arima_inspired,
            "ensemble": self._ensemble_forecast,
        }
        fn = method_map.get(self.method, self._ensemble_forecast)
        predictions = fn(df)

        future_years = np.arange(last_year + 1, last_year + 1 + self.forecast_horizon)
        return pd.DataFrame({
            "Year": future_years,
            "Predicted_Unemployment": [round(p, 4) for p in predictions],
        })

    def forecast_with_confidence(self, df: pd.DataFrame, n_simulations: int = 500) -> pd.DataFrame:
        """
        Monte Carlo simulation for real confidence bands.
        Adds noise based on historical volatility, runs n_simulations, returns
        mean + 10th/90th percentile bounds.
        """
        df = df.copy()
        last_year = int(df["Year"].max())
        future_years = np.arange(last_year + 1, last_year + 1 + self.forecast_horizon)

        # Historical residual std — the true uncertainty measure
        hist_std = float(df["Unemployment_Smoothed"].diff().dropna().std())
        base_preds = np.array(self._ensemble_forecast(df))

        all_sims = []
        rng = np.random.default_rng(42)

        for _ in range(n_simulations):
            # Cumulative noise grows with horizon (uncertainty compounds)
            noise = rng.normal(0, hist_std, self.forecast_horizon)
            cumulative = np.cumsum(noise) * 0.4   # damped so bands aren't too wide
            sim = np.clip(base_preds + cumulative, 0.0, None)
            all_sims.append(sim)

        sims = np.array(all_sims)

        return pd.DataFrame({
            "Year": future_years,
            "Predicted_Unemployment": np.round(base_preds, 4),
            "Lower_80": np.round(np.percentile(sims, 10, axis=0), 4),
            "Upper_80": np.round(np.percentile(sims, 90, axis=0), 4),
            "Lower_95": np.round(np.percentile(sims, 2.5, axis=0), 4),
            "Upper_95": np.round(np.percentile(sims, 97.5, axis=0), 4),
        })
