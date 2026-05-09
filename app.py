"""
app.py
------
StockSARIMA x Prophet — Streamlit dashboard
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from data_pipeline import TICKER_MAP, download_stock_data
from model_utils import (
    ensemble_forecast,
    load_model,
    prophet_future_only,
    sarima_forecast,
    trading_signal,
)
from train import train_all

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StockSARIMA x Prophet",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #000000 !important;
        color: #e0e6f0;
    }

    /* Force black background everywhere */
    .stApp, .main, .block-container {
        background-color: #000000 !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        max-width: 100% !important;
    }

    /* KPI grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #0d0d0d;
        border: 1px solid #1a1a2e;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: .68rem;
        font-weight: 500;
        letter-spacing: .12em;
        color: #5a6a80;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-family: 'Inter', sans-serif;
        font-size: .68rem;
        font-weight: 400;
        color: #5a6a80;
    }

    /* Signal card */
    .signal-card {
        background: #0d0d0d;
        border: 1px solid #1a1a2e;
        border-radius: 10px;
        padding: .75rem 1rem;
        margin-bottom: .5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Performance table */
    .perf-table {
        width: 100%;
        border-collapse: collapse;
    }
    .perf-table th {
        font-family: 'Inter', sans-serif;
        font-size: .62rem;
        font-weight: 600;
        letter-spacing: .12em;
        color: #5a6a80;
        text-transform: uppercase;
        padding: .5rem .6rem;
        text-align: left;
        border-bottom: 1px solid #1a1a2e;
    }
    .perf-table td {
        font-family: 'JetBrains Mono', monospace;
        font-size: .78rem;
        font-weight: 500;
        padding: .55rem .6rem;
        border-bottom: 1px solid #111120;
    }
    .badge {
        display: inline-block;
        padding: .2rem .55rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: .78rem;
        font-weight: 600;
    }
    .badge-arima  { background: #0a1f3a; color: #00d4ff; }
    .badge-prophet{ background: #2a1000; color: #ff7043; }
    .badge-green  { background: #001a12; color: #00e676; }
    .badge-metric { background: #0d1528; color: #7b9cff; }

    .warn-box {
        background: #1a1000;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: .8rem 1rem;
        color: #f59e0b;
        font-size: .8rem;
    }

    div[data-testid="stSidebar"] {
        background: #050505 !important;
        border-right: 1px solid #1a1a2e;
    }
    div[data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif !important;
    }

    /* Pill toggle buttons */
    .stButton > button {
        background: #0d0d0d !important;
        border: 1px solid #1a1a2e !important;
        color: #5a6a80 !important;
        border-radius: 20px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .75rem !important;
        font-weight: 500 !important;
        padding: 0.28rem 0.9rem !important;
        transition: all 0.2s ease !important;
        min-width: 0 !important;
        width: 100% !important;
        letter-spacing: .04em !important;
        white-space: nowrap !important;
    }
    .stButton > button:hover,
    .stButton > button:focus {
        border-color: #00d4ff !important;
        color: #00d4ff !important;
        background: #051520 !important;
    }
    [data-testid="column"] {
        padding-left: 2px !important;
        padding-right: 2px !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        background: #0d0d0d !important;
    }

    /* Section titles */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: .65rem;
        font-weight: 600;
        letter-spacing: .18em;
        color: #5a6a80;
        text-transform: uppercase;
        margin-bottom: .75rem;
    }

    /* Divider */
    hr { border-color: #111120 !important; }

    /* Caption */
    .stCaption, caption {
        font-family: 'Inter', sans-serif !important;
        font-size: .72rem !important;
        color: #5a6a80 !important;
    }

    /* Model panel card */
    .panel-card {
        background: #0d0d0d;
        border: 1px solid #1a1a2e;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
SIGNAL_COLOR = {"BUY": "#00e676", "SELL": "#ff5252", "HOLD": "#90a4ae"}


def fmt_price(val):
    try:
        v = float(val)
        if np.isfinite(v) and v > 0:
            return f"{v:,.0f}"
    except Exception:
        pass
    return "N/A"


def fmt_metric(val, is_pct=False):
    try:
        v = float(val)
        if np.isfinite(v):
            if is_pct:
                return f"{v:.2f}%"
            return f"{v:.4f}"
    except Exception:
        pass
    return "N/A"


def render_signal_card(label, signal, conf, note):
    color = SIGNAL_COLOR.get(signal, "#90a4ae")
    arrow = "▲" if signal == "BUY" else ("▼" if signal == "SELL" else "●")
    html = (
        '<div class="signal-card">'
        f'<div style="background:#111120;border-radius:8px;padding:.4rem .65rem;margin-right:.8rem;">'
        f'<span style="color:{color};font-size:1.1rem;">{arrow}</span>'
        '</div>'
        '<div style="flex:1;">'
        f'<div style="font-family:Inter,sans-serif;font-size:.78rem;font-weight:600;color:{color}">{label}: {signal}</div>'
        f'<div style="font-family:Inter,sans-serif;font-size:.65rem;color:#5a6a80;margin-top:.1rem">{note}</div>'
        '</div>'
        '<div style="text-align:right;">'
        f'<div style="color:{color};font-family:Inter,sans-serif;font-size:.9rem;font-weight:700">{conf:.0f}%</div>'
        '<div style="font-family:Inter,sans-serif;font-size:.6rem;color:#5a6a80">conf.</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<p style="font-family:Inter,sans-serif;font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;">'
        '<span style="color:#00d4ff">Stock</span>'
        '<span style="color:#e0e6f0">SARIMA</span>'
        ' <span style="color:#3a4a5a">×</span> '
        '<span style="color:#ff7043">Prophet</span>'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    ticker_display = st.selectbox(
        "📌 Select Stock",
        options=list(TICKER_MAP.keys()),
        index=0,
    )
    st.markdown("---")
    forecast_days = st.slider("🔭 Forecast Horizon (days)", 1, 7, 7)
    sarima_w = st.slider("⚖️ SARIMA Weight", 0.0, 1.0, 0.4, 0.05)
    prophet_w = round(1.0 - sarima_w, 2)
    st.caption(f"Prophet weight: **{prophet_w}**")
    st.markdown("---")
    force_refresh = st.checkbox("🔄 Force re-download data", value=False)
    retrain_btn = st.button("🚀 Train / Retrain Models", use_container_width=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load(ticker, refresh):
    return download_stock_data(ticker, force_refresh=refresh)


with st.spinner(f"Loading data for {ticker_display} …"):
    try:
        df = cached_load(ticker_display, force_refresh)
        close = df["Close"]
    except Exception as exc:
        st.error(f"❌ Data download failed: {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# Train / Retrain
# ---------------------------------------------------------------------------
if retrain_btn:
    with st.spinner("Training SARIMA + Prophet … this may take a few minutes"):
        try:
            train_all(ticker_display)
            st.success("✅ Models trained and saved successfully!")
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Training failed: {exc}")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
sarima_payload = load_model(ticker_display, "sarima")
prophet_payload = load_model(ticker_display, "prophet")
sarima_ok = sarima_payload is not None
prophet_ok = prophet_payload is not None

if not sarima_ok and not prophet_ok:
    st.markdown(
        '<div class="warn-box">⚠️ No trained models found for this ticker. '
        'Click <strong>Train / Retrain Models</strong> in the sidebar.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------
current_price = float(close.iloc[-1])
last_log_val = float(np.log1p(current_price))

sarima_prices = np.full(forecast_days, np.nan)
sarima_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
if sarima_ok:
    sarima_prices = sarima_forecast(
        sarima_payload["result"],
        steps=forecast_days,
        last_log_value=last_log_val,
    )
    sarima_metrics = sarima_payload.get("metrics", sarima_metrics)

prophet_prices = np.full(forecast_days, np.nan)
prophet_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
if prophet_ok:
    future_df = prophet_future_only(prophet_payload["model"], periods=forecast_days)
    if len(future_df) > 0:
        p_vals = future_df["yhat"].values[:forecast_days]
        prophet_prices[:len(p_vals)] = p_vals
    prophet_metrics = prophet_payload.get("metrics", prophet_metrics)

ensemble_prices = ensemble_forecast(sarima_prices, prophet_prices, sarima_w, prophet_w)

sarima_signal, sarima_conf = trading_signal(current_price, sarima_prices)
prophet_signal, prophet_conf = trading_signal(current_price, prophet_prices)
ensemble_signal, ensemble_conf = trading_signal(current_price, ensemble_prices)

last_date = close.index[-1]
future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

day1_sarima   = sarima_prices[0]   if len(sarima_prices)   and np.isfinite(sarima_prices[0])   else np.nan
day1_prophet  = prophet_prices[0]  if len(prophet_prices)  and np.isfinite(prophet_prices[0])  else np.nan
day1_ensemble = ensemble_prices[0] if len(ensemble_prices) and np.isfinite(ensemble_prices[0]) else np.nan

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown(
        '<span style="font-family:Inter,sans-serif;font-size:1.5rem;font-weight:800;letter-spacing:-0.03em;">'
        '<span style="color:#00d4ff">Stock</span>'
        '<span style="color:#e0e6f0">SARIMA</span>'
        ' <span style="color:#3a4a5a">×</span> '
        '<span style="color:#ff7043">Prophet</span>'
        '</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"🟢 LIVE MODEL  |  {ticker_display}  |  {last_date.date()}")

st.markdown("---")

# ---------------------------------------------------------------------------
# KPI Row — 5 equal cards
# ---------------------------------------------------------------------------
prev = float(close.iloc[-2]) if len(close) > 1 else current_price
delta_pct = (current_price - prev) / prev * 100
delta_color = "#00e676" if delta_pct >= 0 else "#ff5252"
delta_arrow = "▲" if delta_pct >= 0 else "▼"

mae_s = sarima_metrics.get("MAE", np.nan)
rmse_s = sarima_metrics.get("RMSE", np.nan)
mae_p = prophet_metrics.get("MAE", np.nan)
rmse_p = prophet_metrics.get("RMSE", np.nan)

col2_color = "#00d4ff" if np.isfinite(day1_sarima) else "#5a6a80"
col3_color = "#ff7043" if np.isfinite(day1_prophet) else "#5a6a80"
col4_color = "#b39ddb" if np.isfinite(day1_ensemble) else "#5a6a80"
sig_color = SIGNAL_COLOR.get(ensemble_signal, "#90a4ae")

model_order = sarima_payload.get("order", "?") if sarima_ok else "N/A"

kpi_html = f"""
<div class="kpi-grid">
  <div class="metric-card">
    <div class="metric-label">Harga Aktual</div>
    <div class="metric-value" style="color:#00e676">Rp {current_price:,.0f}</div>
    <div class="metric-sub"><span style="color:{delta_color}">{delta_arrow} {abs(delta_pct):.2f}% hari ini</span></div>
  </div>
  <div class="metric-card" style="border-top:2px solid #00d4ff;">
    <div class="metric-label">Prediksi SARIMA {model_order} (D+1)</div>
    <div class="metric-value" style="color:{col2_color}">Rp {fmt_price(day1_sarima)}</div>
    <div class="metric-sub">MAE: {fmt_metric(mae_s)} | RMSE: {fmt_metric(rmse_s)}</div>
  </div>
  <div class="metric-card" style="border-top:2px solid #ff7043;">
    <div class="metric-label">Prediksi Prophet (D+1)</div>
    <div class="metric-value" style="color:{col3_color}">Rp {fmt_price(day1_prophet)}</div>
    <div class="metric-sub">MAE: {fmt_metric(mae_p)} | RMSE: {fmt_metric(rmse_p)}</div>
  </div>
  <div class="metric-card" style="border-top:2px solid #b39ddb;">
    <div class="metric-label">Ensemble Pred. (D+1)</div>
    <div class="metric-value" style="color:{col4_color}">Rp {fmt_price(day1_ensemble)}</div>
    <div class="metric-sub">▲ Bobot {int(sarima_w*100)}% / {int(prophet_w*100)}%</div>
  </div>
  <div class="metric-card" style="border-top:2px solid {sig_color};">
    <div class="metric-label">Sinyal Rekomendasi</div>
    <div class="metric-value" style="color:{sig_color}">{ensemble_signal}</div>
    <div class="metric-sub">Kepercayaan: {ensemble_conf:.0f}%</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main chart + Model performance (side-by-side like the screenshot)
# ---------------------------------------------------------------------------

# Period toggle state — only 1B and 3B
period_days = {"1B": 22, "3B": 66}

if "chart_period" not in st.session_state:
    st.session_state["chart_period"] = "1B"
if "show_sarima" not in st.session_state:
    st.session_state["show_sarima"] = True
if "show_prophet" not in st.session_state:
    st.session_state["show_prophet"] = True

# Hidden Streamlit buttons (invisible, triggered by HTML clicks via JS)
# We use a trick: render real st.buttons but hide them, and use HTML buttons
# that call st.session_state via query params. Instead, use st.columns with
# CSS flex override so they render side-by-side always.
col_chart, col_right = st.columns([7, 3])

with col_chart:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    # ---- Chart title ----
    st.markdown(
        f'<div style="font-family:Inter,sans-serif;font-size:.95rem;font-weight:700;'
        f'color:#e0e6f0;letter-spacing:.01em;margin-bottom:.5rem;">'
        f'{ticker_display} — Prediksi Harga Saham</div>',
        unsafe_allow_html=True,
    )

    # ---- Controls row: model toggles + period — all horizontal via CSS flex ----
    st.markdown("""
    <style>
    /* Force button columns to flex-row (horizontal) */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: none !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 0 3px !important;
    }
    /* Tighten the controls row wrapper */
    .ctrl-row > div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: 4px !important;
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ctrl-row">', unsafe_allow_html=True)
    # All 4 buttons in one flat columns call → guaranteed horizontal
    c_sar, c_pro, _spacer, c_1b, c_3b = st.columns([1.6, 1.6, 6, 0.8, 0.8])
    with c_sar:
        sarima_lbl = "● SARIMA" if st.session_state["show_sarima"] else "○ SARIMA"
        if st.button(sarima_lbl, key="tog_sarima"):
            st.session_state["show_sarima"] = not st.session_state["show_sarima"]
    with c_pro:
        prop_lbl = "● Prophet" if st.session_state["show_prophet"] else "○ Prophet"
        if st.button(prop_lbl, key="tog_prophet"):
            st.session_state["show_prophet"] = not st.session_state["show_prophet"]
    with c_1b:
        if st.button("1B", key="period_1B"):
            st.session_state["chart_period"] = "1B"
    with c_3b:
        if st.button("3B", key="period_3B"):
            st.session_state["chart_period"] = "3B"
    st.markdown('</div>', unsafe_allow_html=True)

    show_sarima  = st.session_state["show_sarima"]
    show_prophet = st.session_state["show_prophet"]

    selected_period = st.session_state["chart_period"]
    lookback = min(period_days[selected_period], len(close))
    hist = close.iloc[-lookback:]

    # ---- Build figure ----
    # All MODEL lines: dashed from history start → through last actual date → forecast
    fig = go.Figure()

    # Actual price — solid green line
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Aktual",
        mode="lines",
        line=dict(color="#a8ff78", width=2),
    ))

    # SARIMA: dashed line from history start → end + forecast (one continuous trace)
    if sarima_ok and show_sarima:
        try:
            fitted_log = sarima_payload["result"].fittedvalues
            log_s = sarima_payload["log_series"]
            aligned = log_s.reindex(fitted_log.index)
            sarima_hist_prices = np.expm1(fitted_log.values)
            sarima_hist_idx = fitted_log.index
            mask = sarima_hist_idx >= hist.index[0]
            sarima_hist_prices_w = sarima_hist_prices[mask]
            sarima_hist_idx_w = sarima_hist_idx[mask]
            all_sarima_x = list(sarima_hist_idx_w) + list(future_dates)
            all_sarima_y = list(sarima_hist_prices_w) + list(sarima_prices)
            fig.add_trace(go.Scatter(
                x=all_sarima_x, y=all_sarima_y,
                name="SARIMA",
                mode="lines",
                line=dict(color="#00d4ff", width=1.8, dash="dash"),
            ))
        except Exception:
            if np.any(np.isfinite(sarima_prices)):
                fig.add_trace(go.Scatter(
                    x=future_dates, y=sarima_prices,
                    name="SARIMA",
                    mode="lines",
                    line=dict(color="#00d4ff", width=1.8, dash="dash"),
                ))

    # Prophet: dashed line from history start → end + forecast (one continuous trace)
    if prophet_ok and show_prophet:
        try:
            p_model = prophet_payload["model"]
            hist_future = p_model.make_future_dataframe(periods=forecast_days)
            p_full_forecast = p_model.predict(hist_future)
            p_hist_df = p_full_forecast[["ds", "yhat"]].copy()
            p_hist_df = p_hist_df[p_hist_df["ds"] >= hist.index[0]]
            fig.add_trace(go.Scatter(
                x=p_hist_df["ds"], y=p_hist_df["yhat"],
                name="Prophet",
                mode="lines",
                line=dict(color="#ff7043", width=1.8, dash="dash"),
            ))
        except Exception:
            if np.any(np.isfinite(prophet_prices)):
                fig.add_trace(go.Scatter(
                    x=future_dates, y=prophet_prices,
                    name="Prophet",
                    mode="lines",
                    line=dict(color="#ff7043", width=1.8, dash="dash"),
                ))

    # Ensemble: dashed from history start → end + forecast (weighted avg of SARIMA+Prophet)
    if np.any(np.isfinite(ensemble_prices)):
        try:
            ens_hist_x = []
            ens_hist_y = []
            if sarima_ok:
                fitted_log_e = sarima_payload["result"].fittedvalues
                log_s_e = sarima_payload["log_series"]
                sarima_h = np.expm1(fitted_log_e.values)
                sarima_h_idx = fitted_log_e.index
                mask_e = sarima_h_idx >= hist.index[0]
                sarima_h_w = sarima_h[mask_e]
                sarima_h_idx_w = sarima_h_idx[mask_e]
                if prophet_ok:
                    p_model_e = prophet_payload["model"]
                    hist_df_e = p_model_e.make_future_dataframe(periods=0)
                    p_fc_e = p_model_e.predict(hist_df_e).set_index("ds")["yhat"]
                    p_reindexed = p_fc_e.reindex(sarima_h_idx_w).values
                    valid_both = np.isfinite(p_reindexed)
                    ens_h_y = np.where(
                        valid_both,
                        sarima_w * sarima_h_w + prophet_w * p_reindexed,
                        sarima_h_w,
                    )
                else:
                    ens_h_y = sarima_h_w
                ens_hist_x = list(sarima_h_idx_w)
                ens_hist_y = list(ens_h_y)
            all_ens_x = ens_hist_x + list(future_dates)
            all_ens_y = ens_hist_y + list(ensemble_prices)
            fig.add_trace(go.Scatter(
                x=all_ens_x, y=all_ens_y,
                name="Ensemble",
                mode="lines",
                line=dict(color="#b39ddb", width=2.2, dash="dash"),
            ))
        except Exception:
            fig.add_trace(go.Scatter(
                x=future_dates, y=ensemble_prices,
                name="Ensemble",
                mode="lines",
                line=dict(color="#b39ddb", width=2.2, dash="dash"),
            ))
        # 95% confidence band (forecast region only)
        upper = ensemble_prices * 1.02
        lower = ensemble_prices * 0.98
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(179,157,219,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Interval 95%",
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(family="Inter, sans-serif", color="#e0e6f0"),
        xaxis=dict(
            gridcolor="#111120",
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=9, color="#5a6a80", family="Inter"),
        ),
        yaxis=dict(
            gridcolor="#111120",
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=9, color="#5a6a80", family="Inter"),
            tickprefix="Rp ",
            tickformat=",",
        ),
        legend=dict(
            orientation="h",
            y=1.06,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, family="Inter"),
        ),
        margin=dict(l=10, r=10, t=45, b=30),
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # ---- Model Performance Panel ----
    st.markdown('<div class="panel-card" style="margin-bottom:12px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Performa Model</div>', unsafe_allow_html=True)

    s_mae  = sarima_metrics.get("MAE", np.nan)
    s_rmse = sarima_metrics.get("RMSE", np.nan)
    s_mape = sarima_metrics.get("MAPE", np.nan)
    s_r2   = sarima_metrics.get("R2", np.nan)
    s_aic  = sarima_payload.get("aic", np.nan) if sarima_ok else np.nan
    s_bic  = sarima_payload.get("bic", np.nan) if sarima_ok else np.nan

    p_mae  = prophet_metrics.get("MAE", np.nan)
    p_rmse = prophet_metrics.get("RMSE", np.nan)
    p_mape = prophet_metrics.get("MAPE", np.nan)
    p_r2   = prophet_metrics.get("R2", np.nan)

    def val_or_dash(v, pct=False):
        try:
            fv = float(v)
            if np.isfinite(fv):
                return f"{fv:.2f}%" if pct else f"{fv:,.0f}" if fv > 10 else f"{fv:.3f}"
        except Exception:
            pass
        return "—"

    def badge(val, cls):
        return f'<span class="badge {cls}">{val}</span>'

    perf_html = f"""
    <table class="perf-table">
      <thead>
        <tr>
          <th>Metrik</th>
          <th>SARIMA</th>
          <th>Prophet</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{badge("MAE","badge-metric")}</td>
          <td>{badge(val_or_dash(s_mae),"badge-arima")}</td>
          <td>{badge(val_or_dash(p_mae),"badge-prophet")}</td>
        </tr>
        <tr>
          <td>{badge("RMSE","badge-metric")}</td>
          <td>{badge(val_or_dash(s_rmse),"badge-arima")}</td>
          <td>{badge(val_or_dash(p_rmse),"badge-prophet")}</td>
        </tr>
        <tr>
          <td>{badge("MAPE","badge-metric")}</td>
          <td>{badge(val_or_dash(s_mape,True),"badge-arima")}</td>
          <td>{badge(val_or_dash(p_mape,True),"badge-prophet")}</td>
        </tr>
        <tr>
          <td>{badge("R²","badge-metric")}</td>
          <td style="color:#00e676;font-family:'JetBrains Mono',monospace;font-size:.78rem">{val_or_dash(s_r2)}</td>
          <td style="color:#00e676;font-family:'JetBrains Mono',monospace;font-size:.78rem">{val_or_dash(p_r2)}</td>
        </tr>
        <tr>
          <td style="color:#5a6a80;font-family:'JetBrains Mono',monospace;font-size:.75rem">AIC</td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:.78rem">{val_or_dash(s_aic)}</td>
          <td style="color:#5a6a80;font-family:'JetBrains Mono',monospace;font-size:.75rem">—</td>
        </tr>
        <tr>
          <td style="color:#5a6a80;font-family:'JetBrains Mono',monospace;font-size:.75rem">BIC</td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:.78rem">{val_or_dash(s_bic)}</td>
          <td style="color:#5a6a80;font-family:'JetBrains Mono',monospace;font-size:.75rem">—</td>
        </tr>
        <tr>
          <td style="color:#5a6a80;font-family:'JetBrains Mono',monospace;font-size:.75rem">Model</td>
          <td>{badge(str(model_order),"badge-arima")}</td>
          <td>{badge("Prophet","badge-prophet")}</td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(perf_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Trading Signals Panel ----
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sinyal Trading</div>', unsafe_allow_html=True)
    render_signal_card(
        "Prophet", prophet_signal, prophet_conf,
        "Trend berdasarkan komponen musiman",
    )
    render_signal_card(
        "SARIMA", sarima_signal, sarima_conf,
        "Parameter AR mendukung arah harga",
    )
    render_signal_card(
        "Ensemble", ensemble_signal, ensemble_conf,
        f"Weighted {int(sarima_w*100)}% / {int(prophet_w*100)}%",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bottom row: residuals | accuracy | forecast table
# ---------------------------------------------------------------------------
st.markdown("---")
col_res, col_acc, col_table = st.columns(3)

with col_res:
    st.markdown('<div class="section-title">📉 Residual Error · SARIMA</div>', unsafe_allow_html=True)
    if sarima_ok:
        try:
            fitted_log = sarima_payload["result"].fittedvalues
            log_s = sarima_payload["log_series"]
            aligned = log_s.reindex(fitted_log.index)
            residuals = np.expm1(aligned.values) - np.expm1(fitted_log.values)
            res_idx = fitted_log.index
            valid_mask = np.isfinite(residuals)
            residuals_clean = residuals[valid_mask]
            res_idx_clean = res_idx[valid_mask]
            tail_res = residuals_clean[-60:]
            tail_idx = res_idx_clean[-60:]
            bar_colors = ["#00d4ff" if v >= 0 else "#ff5252" for v in tail_res]
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=tail_idx,
                y=tail_res,
                marker_color=bar_colors,
                opacity=0.75,
                showlegend=False,
            ))
            fig_res.update_layout(
                template="plotly_dark",
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                margin=dict(l=5, r=5, t=5, b=30),
                height=200,
                font=dict(family="Inter"),
                xaxis=dict(gridcolor="#111120", showgrid=False,
                           tickfont=dict(size=8, color="#5a6a80")),
                yaxis=dict(gridcolor="#111120",
                           tickfont=dict(size=8, color="#5a6a80")),
            )
            st.plotly_chart(fig_res, use_container_width=True)
        except Exception:
            st.caption("Residuals unavailable")
    else:
        st.caption("SARIMA model not loaded")

with col_acc:
    st.markdown('<div class="section-title">🎯 Akurasi Model</div>', unsafe_allow_html=True)

    sarima_mape = sarima_metrics.get("MAPE", np.nan)
    prophet_mape = prophet_metrics.get("MAPE", np.nan)
    sarima_acc = max(0.0, 100.0 - float(sarima_mape)) if np.isfinite(sarima_mape) else 0.0
    prophet_acc = max(0.0, 100.0 - float(prophet_mape)) if np.isfinite(prophet_mape) else 0.0

    st.markdown(
        f'<div style="background:#0a1528;border:1px solid #00d4ff;border-radius:8px;'
        f'padding:.35rem .85rem;display:inline-block;font-size:.7rem;font-family:Inter;margin-bottom:.7rem;">'
        f'Model <span style="color:#00d4ff;font-weight:600">{model_order}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:Inter;font-size:.75rem;font-weight:600;color:#00d4ff;margin-bottom:.2rem;">SARIMA</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(sarima_acc / 100.0, 1.0))
    st.caption(f"{sarima_acc:.1f}%")

    st.markdown(
        '<div style="font-family:Inter;font-size:.75rem;font-weight:600;color:#ff7043;margin-bottom:.2rem;">Prophet</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(prophet_acc / 100.0, 1.0))
    st.caption(f"{prophet_acc:.1f}%")

with col_table:
    st.markdown('<div class="section-title">📅 Forecast 7 Hari</div>', unsafe_allow_html=True)
    rows = []
    for i in range(min(7, forecast_days)):
        rows.append({
            "Hari": f"D+{i+1} {future_dates[i].strftime('%a %d %b')}",
            "SARIMA":   fmt_price(sarima_prices[i]),
            "Prophet":  fmt_price(prophet_prices[i]),
            "Ensemble": fmt_price(ensemble_prices[i]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: Aplikasi ini hanya untuk tujuan edukasi. "
    "Jangan gunakan prediksi ini sebagai satu-satunya dasar keputusan investasi."
)