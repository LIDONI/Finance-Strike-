"""Exploration ARIMA : ACF/PACF, ajustement, résumé et prévision prolongée.

Usage : python -m scripts.explore_arima [--ticker AAPL] [--show]
"""
import argparse

import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.config import ARIMA_EXPLORATION, DEFAULT_TEST_TICKER, OUTPUT_DIR
from src.data_loader import get_series, load_stocks
from src.forecasting.arima import fit_arima


def main(ticker: str, show: bool) -> None:
    cfg = ARIMA_EXPLORATION
    series = get_series(load_stocks(), ticker)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Différenciation pour rendre la série stationnaire, puis ACF / PACF pour choisir p et q
    series_diff = series.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    sm.graphics.tsa.plot_acf(series_diff, lags=cfg["acf_lags"], ax=axes[0])
    sm.graphics.tsa.plot_pacf(series_diff, lags=cfg["acf_lags"], ax=axes[1])
    acf_path = OUTPUT_DIR / f"arima_acf_pacf_{ticker}.png"
    fig.savefig(acf_path, dpi=150, bbox_inches="tight")

    # 2. Ajustement du modèle et résumé
    model_fit = fit_arima(series, cfg["order"])
    print(model_fit.summary())

    # 3. Prévision sur une période prolongée
    end = len(series) + cfg["forecast_steps"] - 1
    predictions = model_fit.predict(start=0, end=end, dynamic=False)

    plt.figure(figsize=(10, 6))
    plt.plot(series, label="Original Series")
    plt.plot(predictions, label="ARIMA Predictions", color="red")
    plt.legend()
    pred_path = OUTPUT_DIR / f"arima_{ticker}.png"
    plt.savefig(pred_path, dpi=150, bbox_inches="tight")
    print(f"Graphiques enregistrés : {acf_path}, {pred_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=DEFAULT_TEST_TICKER)
    parser.add_argument("--show", action="store_true", help="afficher les fenêtres des graphiques")
    args = parser.parse_args()
    main(args.ticker, args.show)
