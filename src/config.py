"""Configuration centralisée du projet (chemins, tickers, hyperparamètres)."""
from pathlib import Path

# --- Chemins -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
STOCKS_CSV = DATA_DIR / "stocks.csv"

# --- Données -----------------------------------------------------------------
TICKERS = ["AAPL", "TSLA", "AMZN", "META", "NFLX", "GOOG", "MCD"]
HISTORY_DAYS = 365  # profondeur d'historique téléchargé
COLUMNS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
DEFAULT_TEST_TICKER = "AAPL"  # ticker utilisé par les scripts d'exploration

# --- Finance -----------------------------------------------------------------
TRADING_DAYS = 252  # jours de bourse par an (annualisation)

# --- Interface ---------------------------------------------------------------
APP_TITLE = "Finance Strike!"
CHART_TYPES = (
    "Séries temporelles",
    "Prix de clôture ajusté",
    "Volatilité (écart-type)",
    "Matrice de corrélation",
    "Variation des prix (%)",
    "Risque vs Retour",
)
MODEL_NONE = "Aucun"
MODEL_CHOICES = (MODEL_NONE, "LSTM")

# --- Prévision ---------------------------------------------------------------
FORECAST_STEPS = 30

ARIMA_ORDER = (1, 1, 1)  # (p, d, q)
ARIMA_MIN_OBS = 60

# LSTM utilisé dans l'application (rapide)
LSTM_APP = dict(seq_length=60, units=50, dense_units=None, epochs=20, batch_size=32)

# LSTM utilisé dans scripts/explore_lstm.py (plus lent, horizon plus long)
LSTM_EXPLORATION = dict(
    seq_length=100, units=50, dense_units=25, epochs=20, batch_size=1,
    train_ratio=0.8, forecast_steps=100,
)

# ARIMA utilisé dans scripts/explore_arima.py
ARIMA_EXPLORATION = dict(order=(1, 1, 1), forecast_steps=100, acf_lags=40)

# --- ngrok -------------------------------------------------------------------
STREAMLIT_PORT = 8501
NGROK_TOKEN_ENV = "NGROK_AUTHTOKEN"
