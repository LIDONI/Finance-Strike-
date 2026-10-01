"""Téléchargement, sauvegarde et chargement des données boursières."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.config import COLUMNS, HISTORY_DAYS, STOCKS_CSV, TICKERS


def download_stocks(tickers=TICKERS, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Télécharge l'historique de chaque ticker via yfinance et renvoie un DataFrame long."""
    import yfinance as yf  # import tardif : inutile si le CSV existe déjà

    today = date.today()
    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    frames = []
    for ticker in tickers:
        data = yf.download(
            ticker, start=start_date, end=end_date,
            progress=False, auto_adjust=False, threads=False,
        )
        if data.empty:
            raise ValueError(f"Aucune donnée reçue pour {ticker}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)  # aplatit le MultiIndex
        data["Ticker"] = ticker
        frames.append(data)

    df = pd.concat(frames).reset_index().sort_values(by=["Ticker", "Date"])
    return df[COLUMNS]


def save_stocks(df: pd.DataFrame, path: Path = STOCKS_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_stocks(path: Path = STOCKS_CSV, auto_download: bool = True) -> pd.DataFrame:
    """Charge le CSV ; le télécharge d'abord s'il n'existe pas (si auto_download)."""
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"{path} introuvable. Lancez : python -m scripts.download_data"
            )
        save_stocks(download_stocks(), path)

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def pivot_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau Date x Ticker des prix de clôture ajustés."""
    return df.pivot(index="Date", columns="Ticker", values="Adj Close")


def get_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Série 'Adj Close' d'un seul ticker, indexée par date."""
    return (
        df[df["Ticker"] == ticker]
        .sort_values("Date")
        .set_index("Date")["Adj Close"]
        .dropna()
    )
