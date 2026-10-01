"""Calculs financiers (rendements, volatilité, corrélation, risque/retour)."""
import pandas as pd

from src.config import TRADING_DAYS


def daily_returns(pivot_df: pd.DataFrame) -> pd.DataFrame:
    return pivot_df.pct_change(fill_method=None).dropna()


def annualized_volatility(pivot_df: pd.DataFrame) -> pd.Series:
    return (
        daily_returns(pivot_df).std().mul(TRADING_DAYS ** 0.5).sort_values(ascending=False)
    )


def correlation_matrix(pivot_df: pd.DataFrame) -> pd.DataFrame:
    return daily_returns(pivot_df).corr()


def price_change_pct(pivot_df: pd.DataFrame) -> pd.Series:
    return (pivot_df.iloc[-1] - pivot_df.iloc[0]) / pivot_df.iloc[0] * 100


def risk_return(pivot_df: pd.DataFrame) -> pd.DataFrame:
    returns = daily_returns(pivot_df)
    return pd.DataFrame(
        {
            "Risk": returns.std() * (TRADING_DAYS ** 0.5),
            "Annualized Return": returns.mean() * TRADING_DAYS,
        }
    )
