"""Point d'entrée Streamlit : streamlit run app.py."""

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import charts
from src.config import (
    APP_TITLE,
    CHART_TYPES,
    FORECAST_STEPS,
    MODEL_CHOICES,
    MODEL_NONE,
)
from src.data_loader import load_stocks, pivot_adj_close
from src.forecasting import get_forecast


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CHEMINS
# ============================================================

# Dossier racine du projet finance_dashboard/
BASE_DIR = Path(__file__).resolve().parent

# Logo situé dans :
# finance_dashboard/assets/Gemini_Generated_Image_v7009jv7009jv700.jpg
LOGO_PATH = (
    BASE_DIR
    / "assets"
    / "Gemini_Generated_Image_v7009jv7009jv700.jpg"
)


# ============================================================
# LOGO
# ============================================================

def get_logo_base64() -> str:
    """Convertit le logo en base64 pour l'afficher dans la sidebar."""

    if not LOGO_PATH.exists():
        return ""

    with LOGO_PATH.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


logo_base64 = get_logo_base64()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       BARRE DU HAUT
       ======================================================== */

    header[data-testid="stHeader"] {
        background-color: #000000 !important;
        height: 70px !important;
    }

    header[data-testid="stHeader"]::before {
        content: "";
        display: block;
        height: 70px;
        background-color: #000000;
    }

    /* Bouton menu / hamburger */
    header[data-testid="stHeader"] button {
        color: #ffffff !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #000000 !important;
    }

    /* Texte général */
    section[data-testid="stSidebar"] {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Titres */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Paragraphes */
    section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] > div {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }


    /* ========================================================
       DATE INPUT
       ======================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stDateInput"] {
        color: #ffffff !important;
    }

    /* Champ de date */
    section[data-testid="stSidebar"]
    div[data-testid="stDateInput"] input {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }

    /* Conteneur du champ */
    section[data-testid="stSidebar"]
    div[data-testid="stDateInput"] div[data-baseweb="input"] {
        background-color: #1f1f1f !important;
        border-color: #444444 !important;
    }

    /* Bouton calendrier */
    section[data-testid="stSidebar"]
    div[data-testid="stDateInput"] button {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }


    /* ========================================================
       LOGO EN BAS DE LA SIDEBAR
       ======================================================== */

    .sidebar-logo {
        position: fixed;
        bottom: 20px;
        left: 0;
        width: 244px;
        text-align: center;
        padding: 10px;
        background-color: #000000;
        z-index: 999;
    }

    .sidebar-logo img {
        width: 180px;
        max-height: 120px;
        object-fit: contain;
        border-radius: 10px;
    }


    /* ========================================================
       CONTENU PRINCIPAL
       ======================================================== */

    .main .block-container {
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOGO SIDEBAR
# ============================================================

if logo_base64:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-logo">
            <img
                src="data:image/jpeg;base64,{logo_base64}"
                alt="Finance Dashboard Logo"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.warning(
        f"Logo introuvable : {LOGO_PATH}"
    )


# ============================================================
# DONNÉES
# ============================================================

@st.cache_data(show_spinner="Chargement des données...")
def get_data() -> pd.DataFrame:
    """Charge les données financières."""
    return load_stocks()


# ============================================================
# PRÉVISION
# ============================================================

@st.cache_data(show_spinner="Calcul de la prévision...")
def cached_forecast(
    model_name: str,
    series: pd.Series,
    steps: int,
):
    """
    Calcule la prévision et met le résultat en cache
    pour éviter de réentraîner le modèle à chaque interaction.
    """
    return get_forecast(model_name, series, steps)


# ============================================================
# APPLICATION
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(APP_TITLE)

    st.sidebar.title("Financial Analysis")

    st.sidebar.markdown(
        "Finance Dashboard!"
    )

    st.markdown(
        "My first finance Dashboard!"
    )


    # --------------------------------------------------------
    # DONNÉES
    # --------------------------------------------------------

    df = get_data()


    # --------------------------------------------------------
    # FILTRES
    # --------------------------------------------------------

    start_date = pd.to_datetime(
        st.sidebar.date_input(
            "Start Date",
            df["Date"].min(),
        )
    )

    end_date = pd.to_datetime(
        st.sidebar.date_input(
            "End Date",
            df["Date"].max(),
        )
    )


    # Vérification des dates
    if start_date > end_date:
        st.error(
            "La date de début doit précéder la date de fin."
        )
        st.stop()


    # Filtrage des données
    df_filtered = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ]


    # Vérification : aucune donnée
    if df_filtered.empty:
        st.warning(
            "Aucune donnée disponible pour cette période."
        )
        st.stop()


    # Transformation pour les graphiques
    pivot_df = pivot_adj_close(df_filtered)


    # --------------------------------------------------------
    # TYPE DE GRAPHIQUE
    # --------------------------------------------------------

    chart_type = st.sidebar.selectbox(
        "Type de graphique",
        CHART_TYPES,
    )


    # --------------------------------------------------------
    # MODÈLE DE PRÉVISION
    # --------------------------------------------------------

    model_choice = st.sidebar.selectbox(
        "Choisissez le modèle de prévision",
        MODEL_CHOICES,
    )


    # --------------------------------------------------------
    # GRAPHIQUE PRINCIPAL
    # --------------------------------------------------------

    if chart_type == "Prix de clôture ajusté":

        fig = charts.adj_close_chart(df_filtered)

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    else:

        chart_function = charts.PLOTLY_CHARTS.get(chart_type)

        if chart_function is None:
            st.error(
                f"Type de graphique inconnu : {chart_type}"
            )
        else:
            st.plotly_chart(
                chart_function(pivot_df),
                use_container_width=True,
            )


    # --------------------------------------------------------
    # PRÉVISION
    # --------------------------------------------------------

    if model_choice != MODEL_NONE:

        st.subheader(
            f"Prévisions basées sur le modèle {model_choice}"
        )


        # Vérification qu'il existe des tickers
        if pivot_df.empty or len(pivot_df.columns) == 0:

            st.warning(
                "Aucun ticker disponible pour effectuer une prévision."
            )

            return


        ticker = st.selectbox(
            "Sélectionnez un ticker",
            pivot_df.columns,
        )


        # Série temporelle du ticker sélectionné
        series = pivot_df[ticker].dropna()


        # Vérification du nombre de données
        if series.empty:

            st.warning(
                f"Aucune donnée disponible pour le ticker {ticker}."
            )

            return


        try:

            forecast, dates = cached_forecast(
                model_choice,
                series,
                FORECAST_STEPS,
            )

        except ValueError as exc:

            st.error(str(exc))

        except Exception as exc:

            st.error(
                f"Une erreur est survenue lors du calcul de la prévision : {exc}"
            )

        else:

            forecast_fig = charts.forecast_chart(
                series,
                forecast,
                dates,
                ticker,
                model_choice,
            )

            st.plotly_chart(
                forecast_fig,
                use_container_width=True,
            )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    main()