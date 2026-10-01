"""Exploration LSTM : entraînement/test sur un ticker et prévision à horizon prolongé.

Usage : python -m scripts.explore_lstm [--ticker AAPL] [--show]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.config import DEFAULT_TEST_TICKER, LSTM_EXPLORATION, OUTPUT_DIR
from src.data_loader import get_series, load_stocks
from src.forecasting.lstm import build_lstm_model, make_sequences, recursive_forecast


def main(ticker: str, show: bool) -> None:
    cfg = LSTM_EXPLORATION
    series = get_series(load_stocks(), ticker)
    seq_length = cfg["seq_length"]

    # Le scaler n'est ajusté que sur la partie entraînement (évite la fuite de données)
    split_point = int(len(series) * cfg["train_ratio"])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(series.iloc[:split_point].values.reshape(-1, 1))
    scaled = scaler.transform(series.values.reshape(-1, 1))

    X, y = make_sequences(scaled, seq_length)
    split_seq = max(1, int(len(X) * cfg["train_ratio"]))
    X_train, X_test = X[:split_seq], X[split_seq:]
    y_train = y[:split_seq]

    model = build_lstm_model(seq_length, cfg["units"], cfg["dense_units"])
    model.fit(X_train, y_train, batch_size=cfg["batch_size"], epochs=cfg["epochs"])

    pred_train = scaler.inverse_transform(model.predict(X_train))
    pred_test = scaler.inverse_transform(model.predict(X_test))

    last_sequence = X_test[-1] if len(X_test) else X_train[-1]
    future = scaler.inverse_transform(
        recursive_forecast(model, last_sequence, cfg["forecast_steps"]).reshape(-1, 1)
    )

    # Positions : la i-ème séquence prédit le point d'indice seq_length + i
    n = len(series)
    train_idx = np.arange(seq_length, seq_length + len(pred_train))
    test_idx = np.arange(seq_length + len(pred_train), seq_length + len(pred_train) + len(pred_test))

    plt.figure(figsize=(12, 6))
    plt.plot(series.values, label="Données Réelles")
    plt.plot(train_idx, pred_train.flatten(), label="Prévisions Entraînement")
    plt.plot(test_idx, pred_test.flatten(), label="Prévisions Test")
    plt.plot(np.arange(n, n + cfg["forecast_steps"]), future.flatten(),
             label="Prévisions Futures", color="green")
    plt.legend()
    plt.title(f"Prévisions de série temporelle avec LSTM ({ticker})")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.grid(True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"lstm_{ticker}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Graphique enregistré : {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=DEFAULT_TEST_TICKER)
    parser.add_argument("--show", action="store_true", help="afficher la fenêtre du graphique")
    args = parser.parse_args()
    main(args.ticker, args.show)
