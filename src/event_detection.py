"""
event_detection.py

Purpose:
    Detect abnormal unemployment changes and label
    economic regimes (stable, shock, recovery).

This module enables scenario-aware forecasting.
"""

import pandas as pd
import numpy as np


class EventDetector:
    def __init__(self, z_threshold: float = 2.0):
        """
        Parameters:
        - z_threshold: sensitivity for shock detection
        """
        self.z_threshold = z_threshold

    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects abnormal unemployment changes using z-score logic.
        """
        df = df.copy()

        # Year-over-year change
        df["YoY_Change"] = df["Unemployment_Rate"].diff()

        mean_change = df["YoY_Change"].mean()
        std_change = df["YoY_Change"].std()

        df["Z_Score"] = (df["YoY_Change"] - mean_change) / std_change

        # Shock detection
        df["Shock_Event"] = df["Z_Score"].abs() > self.z_threshold

        return df

    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Labels each year as:
        - Stable
        - Shock
        - Recovery
        """
        df = df.copy()
        df["Regime"] = "Stable"

        for i in range(len(df)):
            if df.loc[i, "Shock_Event"]:
                df.loc[i, "Regime"] = "Shock"
            elif i > 0 and df.loc[i - 1, "Shock_Event"]:
                df.loc[i, "Regime"] = "Recovery"

        return df

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full event detection pipeline.
        """
        df = self.detect_events(df)
        df = self.label_regimes(df)
        return df
