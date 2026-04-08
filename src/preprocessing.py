"""
preprocessing.py
Cleans and smooths the raw unemployment time series before modelling.
"""
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, smoothing_window: int = 3):
        self.smoothing_window = smoothing_window

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.dropna(subset=["Unemployment_Rate"])
        df = df.sort_values("Year").reset_index(drop=True)

        df = df[df["Unemployment_Rate"] > 0]

        df["Unemployment_Smoothed"] = (
            df["Unemployment_Rate"]
            .rolling(window=self.smoothing_window, min_periods=1, center=True)
            .mean()
        )

        return df
