"""
data_pipeline.py
----------------
Handles downloading, validating, and caching stock data per ticker.
"""

import os
import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# IDX stocks available in the UI
TICKER_MAP = {
    "ADRO - PT Adaro Energy Indonesia Tbk": "ADRO.JK",
    "PTBA - PT Bukit Asam Tbk": "PTBA.JK",
    "MEDC - PT Medco Energi Internasional Tbk": "MEDC.JK",
    "ITMG - PT Indo Tambangraya Megah Tbk": "ITMG.JK",
}


def get_yf_ticker(display_name: str) -> str:
    """
    Resolve a display name or raw ticker to a yfinance-compatible ticker symbol.
    Falls back to appending '.JK' if not found in TICKER_MAP.
    """
    if display_name in TICKER_MAP:
        return TICKER_MAP[display_name]
    # Accept raw symbols like 'ANTM.JK' or 'ANTM'
    if "." in display_name:
        return display_name.upper()
    return display_name.upper() + ".JK"


def download_stock_data(
    ticker_display: str,
    period: str = "2y",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker and persist to data/{ticker}.csv.

    Parameters
    ----------
    ticker_display : str
        Display name (from TICKER_MAP) or raw yfinance symbol.
    period : str
        yfinance period string, e.g. '2y', '5y'.
    force_refresh : bool
        Re-download even if local cache exists.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with DatetimeIndex and at minimum a 'Close' column.
    """
    yf_ticker = get_yf_ticker(ticker_display)
    safe_name = yf_ticker.replace(".", "_")
    csv_path = os.path.join(DATA_DIR, f"{safe_name}.csv")

    if os.path.exists(csv_path) and not force_refresh:
        logger.info("Loading cached data for %s from %s", yf_ticker, csv_path)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return _validate_and_clean(df, yf_ticker)

    logger.info("Downloading data for %s …", yf_ticker)
    raw = yf.download(yf_ticker, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker '{yf_ticker}'. "
            "Check the symbol and your internet connection."
        )

    df = _validate_and_clean(raw, yf_ticker)
    df.to_csv(csv_path)
    logger.info("Saved %d rows to %s", len(df), csv_path)
    return df


def load_stock_data(ticker_display: str) -> pd.DataFrame:
    """
    Load data from local cache; download if not present.
    """
    return download_stock_data(ticker_display, force_refresh=False)


def _validate_and_clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Internal helper: ensure index is DatetimeIndex, sort, drop duplicates,
    fill small gaps, and verify 'Close' column exists with sensible values.
    """
    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    required_cols = {"Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Data for {ticker} is missing columns: {missing}")

    # Drop rows where Close is NaN or <= 0
    df = df[df["Close"].notna() & (df["Close"] > 0)]

    # Forward-fill small internal gaps (weekend / holiday carry-over)
    df = df.asfreq("B").ffill()  # business-day frequency
    df = df[df["Close"].notna()]  # drop leading NaN rows

    if len(df) < 30:
        raise ValueError(
            f"Insufficient data for {ticker}: only {len(df)} rows after cleaning."
        )

    logger.info(
        "Clean data for %s: %d rows, %s → %s",
        ticker,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def get_close_series(ticker_display: str) -> pd.Series:
    """
    Convenience function: return the Close price Series for a ticker.
    """
    df = load_stock_data(ticker_display)
    return df["Close"].rename(ticker_display)
