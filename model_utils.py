"""
model_utils.py
--------------
Model loading, saving, inference utilities for SARIMA and Prophet.
All SARIMA work is done on log-transformed data; predictions are
exponentiated back to price scale to eliminate overflow.
"""

import os
import logging
import pickle
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _model_path(ticker_display: str, model_type: str) -> str:
    safe = ticker_display.replace(" ", "_").replace("-", "_").replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_{model_type}.pkl")


def save_model(model, ticker_display: str, model_type: str) -> str:
    path = _model_path(ticker_display, model_type)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s model → %s", model_type, path)
    return path


def load_model(ticker_display: str, model_type: str):
    """Load a persisted model. Returns None if file not found."""
    path = _model_path(ticker_display, model_type)
    if not os.path.exists(path):
        logger.warning("Model file not found: %s", path)
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def model_exists(ticker_display: str, model_type: str) -> bool:
    return os.path.exists(_model_path(ticker_display, model_type))


# ---------------------------------------------------------------------------
# SARIMA inference
# ---------------------------------------------------------------------------

def sarima_forecast(
    sarima_result,
    steps: int = 7,
    last_log_value: float = None,
) -> np.ndarray:
    """
    Generate out-of-sample forecasts from a fitted SARIMA model.

    The model is trained on log(Close), so we exponentiate the output.
    A sanity clamp prevents extreme values even if the model drifts.

    Parameters
    ----------
    sarima_result : SARIMAXResults
        Fitted statsmodels SARIMAX result object.
    steps : int
        Number of steps ahead.
    last_log_value : float
        log(last actual close price) used for drift clamping.

    Returns
    -------
    np.ndarray of shape (steps,) in original price scale.
    """
    try:
        forecast_log = sarima_result.forecast(steps=steps)
        forecast_log = np.array(forecast_log, dtype=float)

        # Clamp in log space: no more than ±50% price move per step
        if last_log_value is not None:
            max_drift = np.log(1.5)  # 50 % per step
            lo = last_log_value - max_drift * steps
            hi = last_log_value + max_drift * steps
            forecast_log = np.clip(forecast_log, lo, hi)

        prices = np.expm1(forecast_log)  # inverse of log1p
        prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
        return prices
    except Exception as exc:
        logger.error("SARIMA forecast failed: %s", exc)
        return np.full(steps, np.nan)


def sarima_in_sample(sarima_result, log_series: pd.Series) -> pd.Series:
    """
    Return in-sample fitted values in original price scale.
    """
    try:
        fitted_log = sarima_result.fittedvalues
        prices = np.expm1(fitted_log)
        prices = prices.where(prices > 0)
        prices.index = log_series.index[-len(prices):]
        return prices
    except Exception as exc:
        logger.error("SARIMA in-sample failed: %s", exc)
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Prophet inference
# ---------------------------------------------------------------------------

def prophet_forecast(prophet_model, periods: int = 7) -> pd.DataFrame:
    """
    Extend Prophet's future dataframe and predict.

    Returns
    -------
    pd.DataFrame with columns [ds, yhat, yhat_lower, yhat_upper].
    """
    try:
        future = prophet_model.make_future_dataframe(periods=periods, freq="B")
        forecast = prophet_model.predict(future)
        forecast["yhat"] = np.where(forecast["yhat"] > 0, forecast["yhat"], np.nan)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as exc:
        logger.error("Prophet forecast failed: %s", exc)
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])


def prophet_future_only(prophet_model, periods: int = 7) -> pd.DataFrame:
    """Return only the *future* rows (beyond training data)."""
    full = prophet_forecast(prophet_model, periods)
    if full.empty:
        return full
    cutoff = prophet_model.history["ds"].max()
    return full[full["ds"] > cutoff].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_forecast(
    sarima_prices: np.ndarray,
    prophet_prices: np.ndarray,
    sarima_weight: float = 0.4,
    prophet_weight: float = 0.6,
) -> np.ndarray:
    """
    Weighted average ensemble, skipping NaN values gracefully.

    If one model produces NaN for a step, the other model's value is used
    with full weight. If both are NaN, the result is NaN.
    """
    sarima_arr = np.array(sarima_prices, dtype=float)
    prophet_arr = np.array(prophet_prices, dtype=float)

    assert len(sarima_arr) == len(prophet_arr), "Forecast arrays must have equal length"

    result = np.full(len(sarima_arr), np.nan)
    for i in range(len(sarima_arr)):
        s_ok = np.isfinite(sarima_arr[i]) and sarima_arr[i] > 0
        p_ok = np.isfinite(prophet_arr[i]) and prophet_arr[i] > 0
        if s_ok and p_ok:
            result[i] = sarima_weight * sarima_arr[i] + prophet_weight * prophet_arr[i]
        elif p_ok:
            result[i] = prophet_arr[i]
        elif s_ok:
            result[i] = sarima_arr[i]
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    actual: pd.Series, predicted: pd.Series
) -> Tuple[float, float, float]:
    """
    Compute MAE, RMSE, MAPE on aligned series.

    Returns
    -------
    (mae, rmse, mape)  — all as floats; MAPE is a percentage (0-100).
    """
    actual = actual.dropna()
    predicted = predicted.reindex(actual.index).dropna()
    common_index = actual.index.intersection(predicted.index)
    a = actual.loc[common_index]
    p = predicted.loc[common_index]

    if len(a) == 0:
        return np.nan, np.nan, np.nan

    mae = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    # MAPE — guard against zero actuals
    mask = a != 0
    mape = (np.abs((a[mask] - p[mask]) / a[mask]).mean() * 100) if mask.any() else np.nan
    return float(mae), float(rmse), float(mape)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def trading_signal(
    current_price: float,
    forecast_prices: np.ndarray,
    threshold_pct: float = 0.5,
) -> Tuple[str, float]:
    """
    Simple directional signal based on forecast mean vs current price.

    Returns
    -------
    (signal, confidence_pct)  where signal ∈ {'BUY', 'SELL', 'HOLD'}.
    """
    valid = forecast_prices[np.isfinite(forecast_prices) & (forecast_prices > 0)]
    if len(valid) == 0 or current_price <= 0:
        return "HOLD", 0.0

    mean_fc = float(np.mean(valid))
    pct_change = (mean_fc - current_price) / current_price * 100

    if pct_change > threshold_pct:
        signal = "BUY"
        confidence = min(abs(pct_change) * 10, 95.0)
    elif pct_change < -threshold_pct:
        signal = "SELL"
        confidence = min(abs(pct_change) * 10, 95.0)
    else:
        signal = "HOLD"
        confidence = 50.0

    return signal, round(confidence, 1)
