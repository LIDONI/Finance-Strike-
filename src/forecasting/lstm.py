"""Prévision LSTM (TensorFlow/Keras importés à la demande pour accélérer le démarrage)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import LSTM_APP


def make_sequences(data: np.ndarray, seq_length: int):
    """Découpe une série normalisée en fenêtres (X) et cibles (y)."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length: int, units: int = 50, dense_units: int | None = None):
    from keras.layers import LSTM, Dense, Input
    from keras.models import Sequential

    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units))
    if dense_units:
        model.add(Dense(dense_units))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def recursive_forecast(model, last_sequence: np.ndarray, steps: int) -> np.ndarray:
    """Prévision récursive : chaque prédiction est réinjectée dans la fenêtre. Renvoie des valeurs normalisées."""
    seq_length = len(last_sequence)
    seq = last_sequence.copy()
    preds = []
    for _ in range(steps):
        pred = model.predict(seq.reshape(1, seq_length, 1), verbose=0)
        preds.append(pred[0, 0])
        seq = np.append(seq[1:], pred).reshape(seq_length, 1)
    return np.array(preds)


def lstm_forecast(series: pd.Series, steps: int, **params) -> np.ndarray:
    cfg = {**LSTM_APP, **params}
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = make_sequences(scaled, cfg["seq_length"])
    model = build_lstm_model(cfg["seq_length"], cfg["units"], cfg["dense_units"])
    model.fit(X, y, epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=0)

    forecast_scaled = recursive_forecast(model, scaled[-cfg["seq_length"]:], steps)
    return scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
