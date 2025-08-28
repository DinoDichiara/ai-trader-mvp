import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from utils import compute_rsi, vector_backtest, metrics_report

st.set_page_config(page_title="AI Trading MVP", layout="wide")

st.title("AI-Powered Trading MVP (EOD)")

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Symbol", value="AAPL")
    start = st.date_input("Start", pd.to_datetime("2017-01-01"))
    end = st.date_input("End", pd.to_datetime("today"))
    fast = st.number_input("Fast SMA (days)", min_value=2, max_value=200, value=10)
    slow = st.number_input("Slow SMA (days)", min_value=3, max_value=400, value=30)
    rsi_len = st.number_input("RSI length", min_value=2, max_value=50, value=14)
    rsi_low = st.number_input("RSI oversold ≤", min_value=0, max_value=100, value=35)
    rsi_high = st.number_input("RSI overbought ≥", min_value=0, max_value=100, value=65)
    capital = st.number_input("Initial capital ($)", min_value=1000, value=10000, step=100)
    risk_free = st.number_input("Risk-free rate (annual, %)", 0.0, 10.0, 0.0, step=0.1)
    run_btn = st.button("Run Backtest")

st.markdown("This MVP focuses on **explainable signals**: SMA crossover + RSI filter. "
            "Next iterations will add ML models and paper trading (Alpaca).")

@st.cache_data(show_spinner=False)
def load_data(sym, start, end):
    df = yf.download(sym, start=start, end=end, auto_adjust=True)
    df = df.dropna()
    return df

if run_btn:
    try:
        data = load_data(ticker, start, end)
        if data.empty:
            st.error("No data returned. Check the ticker or dates.")
            st.stop()

        df = data.copy()
        df["SMA_fast"] = df["Close"].rolling(int(fast)).mean()
        df["SMA_slow"] = df["Close"].rolling(int(slow)).mean()
        df["RSI"] = compute_rsi(df["Close"], int(rsi_len))

        # Explainable rule: long when fast>slow and RSI in a healthy range
        df["long_signal"] = (df["SMA_fast"] > df["SMA_slow"]) & (df["RSI"].between(rsi_low, rsi_high))
        df["position"] = np.where(df["long_signal"], 1, 0)  # long/flat

        close_series = df["Close"].squeeze()           # <- asegurar 1D
        pos_series = df["position"].squeeze()

        bt = vector_backtest(close_series, pos_series, rf=risk_free/100.0)
        rep = metrics_report(bt["equity_curve"].squeeze(),
                     bt["returns"].squeeze(),
                     bt["rf_series"].squeeze())

        c1, c2 = st.columns([2, 1], gap="large")

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_fast"].squeeze(), name=f"SMA {fast}"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_slow"].squeeze(), name=f"SMA {slow}"))

            # Mark entries/exits
            entries = df.index[(df["position"].diff() == 1)]
            exits = df.index[(df["position"].diff() == -1)]
            fig.add_trace(go.Scatter(x=entries, y=df.loc[entries, "Close"],
                                     mode="markers", name="Entry", marker_symbol="triangle-up", marker_size=10))
            fig.add_trace(go.Scatter(x=exits, y=df.loc[exits, "Close"],
                                     mode="markers", name="Exit", marker_symbol="triangle-down", marker_size=10))
            fig.update_layout(title=f"{ticker} Price & Signals", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["equity_curve"].squeeze(), name="Equity Curve"))
            fig_eq.update_layout(title="Equity Curve", legend=dict(orientation="h"))
            st.plotly_chart(fig_eq, use_container_width=True)

        with c2:
            st.subheader("Performance")
            st.metric("Total Return", f"{rep['total_return_pct']:.2f}%")
            st.metric("CAGR", f"{rep['cagr_pct']:.2f}%")
            st.metric("Max Drawdown", f"{rep['max_dd_pct']:.2f}%")
            st.metric("Sharpe (annualized)", f"{rep['sharpe']:.2f}")

            st.divider()
            st.subheader("Why these signals?")
            st.markdown(
                f"""
- **SMA fast vs. slow**: captures short-term momentum relative to trend.  
- **RSI [{rsi_len}] in [{rsi_low}, {rsi_high}]**: avoids buying overextended or weak momentum zones.  
- **Position**: 1 when both conditions hold, else 0.  
                """
            )

        st.success("Backtest completed.")

        with st.expander("Data Preview"):
            st.dataframe(df.tail(200))

        with st.expander("Next: Add ML & Paper Trading"):
            st.markdown("""
**Planned hooks:**
- Add GradientBoostingClassifier (features: returns lags, RSI, SMA slopes, ATR).
- Walk-forward validation (TimeSeriesSplit).
- Paper trading via Alpaca (env vars: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`).
- Risk controls: position sizing by volatility target, stop-loss ATR.
""")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Set parameters in the sidebar and click **Run Backtest**.")
