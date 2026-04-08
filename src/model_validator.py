"""
model_validator.py
Validation metrics for the forecasting model: MAE, MAPE, RMSE, R², Directional Accuracy.
"""
import pandas as pd
import numpy as np


class ModelValidator:

    @staticmethod
    def get_validation_report(test_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
        """
        Compares actual test values vs forecast values.
        test_df must have: Year, Unemployment_Rate
        forecast_df must have: Year, Predicted_Unemployment
        """
        merged = pd.merge(test_df, forecast_df, on="Year", how="inner")

        if merged.empty:
            return {
                "mae": None, "mape": None, "rmse": None,
                "r2": None, "directional_accuracy": None,
                "forecast_bias": None,
                "detail": [],
            }

        actual = merged["Unemployment_Rate"].values
        predicted = merged["Predicted_Unemployment"].values
        errors = predicted - actual

        mae = float(np.abs(errors).mean())
        rmse = float(np.sqrt((errors ** 2).mean()))

        non_zero = actual.copy()
        non_zero[non_zero == 0] = np.nan
        mape = float(np.nanmean(np.abs(errors / non_zero) * 100))

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        if len(actual) > 1:
            actual_dir = np.diff(actual) > 0
            pred_dir = np.diff(predicted) > 0
            directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
        else:
            directional_accuracy = None

        forecast_bias = float(errors.mean())

        detail = merged.rename(columns={
            "Unemployment_Rate": "Actual",
            "Predicted_Unemployment": "Predicted",
        })[["Year", "Actual", "Predicted"]].round(3).to_dict(orient="records")

        return {
            "mae": round(mae, 3),
            "mape": round(mape, 2) if not np.isnan(mape) else None,
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 1) if directional_accuracy is not None else None,
            "forecast_bias": round(forecast_bias, 3),
            "detail": detail,
        }
