"""Modèles de prévision et point d'entrée unique `get_forecast`."""
import pandas as pd

from src.forecasting.arima import arima_forecast


def get_forecast(model_name: str, series: pd.Series, steps: int):
    """Renvoie (valeurs prévues, dates de bourse futures)."""
    if model_name == "ARIMA":
        forecast = arima_forecast(series, steps)
    elif model_name == "LSTM":
        from src.forecasting.lstm import lstm_forecast  # import tardif (TensorFlow)
        forecast = lstm_forecast(series, steps)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    # Les dates futures commencent le jour ouvré suivant le dernier point historique
    dates = pd.bdate_range(start=series.index[-1] + pd.offsets.BDay(1), periods=len(forecast))
    return forecast, dates
