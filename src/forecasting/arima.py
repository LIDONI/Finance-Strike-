"""Prévision ARIMA."""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.config import ARIMA_MIN_OBS, ARIMA_ORDER


def fit_arima(series: pd.Series, order=ARIMA_ORDER):
    return ARIMA(series, order=order).fit()


def arima_forecast(series: pd.Series, steps: int, order=ARIMA_ORDER) -> pd.Series:
    series = series.dropna()
    if len(series) < ARIMA_MIN_OBS:
        raise ValueError(f"Au moins {ARIMA_MIN_OBS} observations sont nécessaires pour ARIMA")
    return fit_arima(series, order).forecast(steps=steps)
