"""
train.py
--------
Training pipeline for ARIMA and Prophet models.

Key design decisions:
  - ARIMA is trained on log1p(Close) → predictions are expm1'd back.
  - auto_arima uses an expanded (but not excessive) search space since we
    only have 4 stocks to tune — more options, manageable runtime.
  - Prophet trains on the raw Close price.
  - Training data is limited to the LAST 6 MONTHS (≈130 business days)
    so the model stays fresh and captures recent market regime.
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_pipeline import get_close_series, load_stock_data
from model_utils import save_model, compute_metrics

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training window — last 6 months of business days
# ---------------------------------------------------------------------------
TRAIN_WINDOW_DAYS = 130   # ≈ 6 calendar months of trading days


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def train_sarima(
    ticker_display: str,
    seasonal_period: int = 5,           # 5 trading days = 1 week
    use_auto_arima: bool = True,
    order: Tuple[int, int, int] = (2, 1, 2),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 5),
    test_size: int = 20,                # keep test_size small to leave more for training
) -> dict:
    """
    Fit an ARIMA/SARIMA model on log-transformed Close prices.

    Uses the last TRAIN_WINDOW_DAYS business days of data so the model
    reflects the most recent 6-month market regime.

    auto_arima search space is expanded (max_p/q=4, max_P/Q=2, max_d=2)
    to give better fits across 4 stocks without becoming prohibitively slow.
    """
    full_series = get_close_series(ticker_display)

    # ── Limit to last 6 months ────────────────────────────────────────────
    series = full_series.iloc[-TRAIN_WINDOW_DAYS:] if len(full_series) > TRAIN_WINDOW_DAYS else full_series
    logger.info(
        "ARIMA training window for %s: %d rows (%s → %s)",
        ticker_display, len(series),
        series.index[0].date(), series.index[-1].date(),
    )

    # Log-transform
    log_series = np.log1p(series)

    train = log_series.iloc[:-test_size] if test_size > 0 else log_series
    test  = log_series.iloc[-test_size:]  if test_size > 0 else pd.Series(dtype=float)

    if use_auto_arima:
        logger.info("Running auto_arima for %s …", ticker_display)
        try:
            auto_model = auto_arima(
                train,
                # Non-seasonal AR / MA: wider search (0–4)
                start_p=0, max_p=4,
                start_q=0, max_q=4,
                d=None, max_d=2,
                # Seasonal AR / MA: slightly wider (0–2)
                start_P=0, max_P=2,
                start_Q=0, max_Q=2,
                D=None, max_D=1,
                m=seasonal_period,
                seasonal=True,
                # Use BIC to penalise over-parameterisation
                information_criterion="bic",
                # Parallel candidate evaluation (safe for 4-stock setup)
                stepwise=False,          # exhaustive search — OK for small data window
                n_fits=50,               # cap total candidates evaluated
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
                n_jobs=-1,               # use all cores
            )
            best_order = auto_model.order
            best_seasonal_order = auto_model.seasonal_order
            logger.info(
                "auto_arima selected order=%s seasonal_order=%s",
                best_order, best_seasonal_order,
            )
        except Exception as exc:
            logger.warning("auto_arima failed (%s). Falling back to manual order.", exc)
            best_order = order
            best_seasonal_order = seasonal_order
    else:
        best_order = order
        best_seasonal_order = seasonal_order

    logger.info(
        "Fitting SARIMAX(%s)(%s) for %s …",
        best_order, best_seasonal_order, ticker_display,
    )
    model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=300, method="lbfgs")

    # ── Metrics on held-out test set ──────────────────────────────────────
    mae, rmse, mape = np.nan, np.nan, np.nan
    if len(test) > 0:
        try:
            fc_log = result.forecast(steps=len(test))
            fc_prices = np.expm1(np.array(fc_log, dtype=float))
            actual_prices = np.expm1(test.values)
            fc_prices = np.clip(fc_prices, 0, actual_prices.max() * 10)
            fc_series = pd.Series(fc_prices, index=test.index)
            actual_series = pd.Series(actual_prices, index=test.index)
            mae, rmse, mape = compute_metrics(actual_series, fc_series)
        except Exception as exc:
            logger.warning("ARIMA metric calculation failed: %s", exc)

    payload = {
        "result":         result,
        "log_series":     log_series,
        "order":          best_order,
        "seasonal_order": best_seasonal_order,
        "metrics":        {"MAE": mae, "RMSE": rmse, "MAPE": mape},
    }
    save_model(payload, ticker_display, "sarima")
    logger.info("ARIMA MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%", mae, rmse, mape)
    return payload


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def train_prophet(
    ticker_display: str,
    test_size: int = 20,
    # Expanded hyperparameter options — reasonable range, not exhaustive
    changepoint_prior_scale: float = 0.1,    # more flexible trend than default 0.05
    seasonality_prior_scale: float = 15.0,   # slightly stronger seasonal signal
    seasonality_mode: str = "multiplicative", # better for IDX stocks with vol scaling
) -> dict:
    """
    Fit a Prophet model on raw Close prices.

    Training data is limited to the last 6 months (TRAIN_WINDOW_DAYS).
    Hyperparameters are tuned for 4-stock IDX regime: multiplicative
    seasonality, more flexible changepoints, stronger seasonal prior.
    """
    df = load_stock_data(ticker_display).reset_index()

    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_df = prophet_df.dropna(subset=["y"])
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    # ── Limit to last 6 months ────────────────────────────────────────────
    if len(prophet_df) > TRAIN_WINDOW_DAYS:
        prophet_df = prophet_df.iloc[-TRAIN_WINDOW_DAYS:].reset_index(drop=True)
    logger.info(
        "Prophet training window for %s: %d rows (%s → %s)",
        ticker_display, len(prophet_df),
        prophet_df["ds"].iloc[0].date(), prophet_df["ds"].iloc[-1].date(),
    )

    train_df = prophet_df.iloc[:-test_size] if test_size > 0 else prophet_df
    test_df  = prophet_df.iloc[-test_size:]  if test_size > 0 else pd.DataFrame()

    logger.info("Fitting Prophet for %s (%d rows) …", ticker_display, len(train_df))
    m = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        n_changepoints=25,               # default 25; explicit for clarity
    )
    # Add monthly seasonality — useful for IDX monthly cycles
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(train_df)

    # ── Metrics ───────────────────────────────────────────────────────────
    mae, rmse, mape = np.nan, np.nan, np.nan
    if len(test_df) > 0:
        try:
            future = m.make_future_dataframe(periods=len(test_df), freq="B")
            forecast = m.predict(future)
            pred_test = (
                forecast[["ds", "yhat"]]
                .merge(test_df[["ds", "y"]], on="ds", how="inner")
            )
            if len(pred_test) > 0:
                fc_series = pd.Series(pred_test["yhat"].values, index=pred_test["ds"])
                actual_series = pd.Series(pred_test["y"].values,  index=pred_test["ds"])
                mae, rmse, mape = compute_metrics(actual_series, fc_series)
        except Exception as exc:
            logger.warning("Prophet metric calculation failed: %s", exc)

    payload = {
        "model":   m,
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape},
    }
    save_model(payload, ticker_display, "prophet")
    logger.info("Prophet MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%", mae, rmse, mape)
    return payload


# ---------------------------------------------------------------------------
# Convenience: train both models for a ticker
# ---------------------------------------------------------------------------

def train_all(ticker_display: str) -> dict:
    """Train and persist both ARIMA and Prophet for the given ticker."""
    logger.info("=" * 60)
    logger.info("Training all models for: %s", ticker_display)
    logger.info("=" * 60)
    sarima_result  = train_sarima(ticker_display)
    prophet_result = train_prophet(ticker_display)
    return {"sarima": sarima_result, "prophet": prophet_result}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data_pipeline import TICKER_MAP

    if len(sys.argv) > 1:
        target = " ".join(sys.argv[1:])
        if target not in TICKER_MAP:
            matches = [k for k in TICKER_MAP if k.startswith(target)]
            if matches:
                target = matches[0]
        train_all(target)
    else:
        for name in TICKER_MAP:
            try:
                train_all(name)
            except Exception as e:
                logger.error("Failed to train %s: %s", name, e)
