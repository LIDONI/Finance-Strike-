"""Télécharge les données boursières et les enregistre dans data/stocks.csv.

Usage : python -m scripts.download_data
"""
from src.data_loader import download_stocks, save_stocks


def main() -> None:
    df = download_stocks()
    path = save_stocks(df)
    print(f"{len(df)} lignes enregistrées dans {path}")
    print(df.head())


if __name__ == "__main__":
    main()
