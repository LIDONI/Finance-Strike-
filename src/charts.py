"""Fabriques de graphiques (Plotly / Matplotlib). Aucune dépendance à Streamlit."""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src import metrics


def time_series_chart(pivot_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in pivot_df.columns:
        fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[column], name=column))
    fig.update_layout(
        title_text="Affichage des prix de clôture dans le temps",
        xaxis_title="Date", yaxis_title="Prix de fermeture",
        legend_title="Ticker", showlegend=True,
    )
    return fig


def adj_close_chart(df_filtered: pd.DataFrame):
    """Graphique Seaborn/Matplotlib des prix ajustés (renvoie la Figure)."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.lineplot(data=df_filtered, x="Date", y="Adj Close", hue="Ticker", marker="o", ax=ax)
    ax.set_title("Prix de clôture ajusté dans le temps", fontsize=15)
    ax.set_xlabel("Date", fontsize=15)
    ax.set_ylabel("Prix de clôture ajusté", fontsize=14)
    ax.legend(title="Ticker", title_fontsize="12", fontsize="10")
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)
    return fig


def volatility_chart(pivot_df: pd.DataFrame) -> go.Figure:
    vol = metrics.annualized_volatility(pivot_df)
    title = "Volatilité des prix de clôture (écart-type)"
    fig = px.bar(vol, x=vol.index, y=vol.values,
                 labels={"y": "Standard Deviation", "x": "Ticker"}, title=title)
    fig.update_layout(title={"text": f"<b>{title}</b>", "x": 0.5, "xanchor": "center"})
    return fig


def correlation_chart(pivot_df: pd.DataFrame) -> go.Figure:
    corr = metrics.correlation_matrix(pivot_df)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="blues", colorbar=dict(title="Correlation"),
        text=corr.values, texttemplate="%{text:.2f}", hoverinfo="z",
    ))
    fig.update_layout(title="Correlation Matrix of Closing Prices",
                      xaxis_title="Ticker", yaxis_title="Ticker")
    return fig


def price_change_chart(pivot_df: pd.DataFrame) -> go.Figure:
    change = metrics.price_change_pct(pivot_df)
    return px.bar(change, x=change.index, y=change.values,
                  labels={"y": "Percentage Change (%)", "x": "Ticker"},
                  title="Pourcentage de variation des prix de clôture")


def risk_return_chart(pivot_df: pd.DataFrame) -> go.Figure:
    rr = metrics.risk_return(pivot_df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rr["Risk"], y=rr["Annualized Return"], mode="markers+text",
        text=rr.index, textposition="top center", marker=dict(size=10),
    ))
    fig.update_layout(title="Risk vs. Return Analysis",
                      xaxis_title="Risk (Standard Deviation)",
                      yaxis_title="Annualized Return", showlegend=False)
    return fig


def forecast_chart(history: pd.Series, forecast, forecast_dates, ticker: str, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name=f"Historique {ticker}"))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name=f"Prévision {model_name}"))
    fig.update_layout(title=f"Prévisions de prix pour {ticker} avec {model_name}",
                      xaxis_title="Date", yaxis_title="Prix de clôture", showlegend=True)
    return fig


# Dispatch : libellé du menu -> fonction de rendu Plotly (les graphiques Matplotlib sont gérés à part)
PLOTLY_CHARTS = {
    "Séries temporelles": time_series_chart,
    "Volatilité (écart-type)": volatility_chart,
    "Matrice de corrélation": correlation_chart,
    "Variation des prix (%)": price_change_chart,
    "Risque vs Retour": risk_return_chart,
}
