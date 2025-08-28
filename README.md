# AI Trading MVP (macOS) 🚀📈

A friendly, explainable, end-to-end **trading research** app you can run on your Mac.
Fetch EOD data, generate simple signals (SMA + RSI), run fast backtests, visualize performance, and get ready to plug in ML models and paper trading later.

---

## Highlights ✨

* **Explainable signals**: SMA crossover + RSI filter, with a “Why these signals?” panel
* **Vector backtests**: equity curve, total return, CAGR, max drawdown, Sharpe
* **Beautiful dashboard**: Streamlit + interactive Plotly charts
* **Ready to grow**: hooks for ML (walk-forward) and **Alpaca** paper trading
* **Mac-friendly**: simple `venv` + `pip` flow

---

## Tech Stack 🧰

* **Python 3.11**, **Streamlit**, **pandas**, **numpy**, **yfinance**, **plotly**
* (Optional next steps): `scikit-learn`, `alpaca-py`, `ccxt`

---

## Project Structure 📁

```
ai-trader-mvp/
├─ requirements.txt
├─ app.py           # Streamlit dashboard
└─ utils.py         # RSI, vector backtest, metrics
```

---

## Quickstart (macOS) ⚡

```bash
# 1) Clone (or create the folder and add files)
git clone <your-repo-url> ai-trader-mvp
cd ai-trader-mvp

# 2) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Open the local URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

---

## How to Use 🔎

1. Enter a **symbol** (e.g., `AAPL` or `BTC-USD`) and date range.
2. Adjust **SMA/RSI** parameters on the sidebar.
3. Click **Run Backtest**.
4. Review charts (price + SMAs + entries/exits) and the **Equity Curve**.
5. Check the **Performance** metrics and the **Why these signals?** explainer.
6. Iterate parameters, compare outcomes, and take notes.

> Tip: Start with equities like `AAPL` for clean data, then try `BTC-USD`.

---

## How It Works 🧠

* **Data**: pulls EOD prices via `yfinance` (adjusted close).
* **Signals**: Long when `SMA_fast > SMA_slow` **and** `RSI_low ≤ RSI ≤ RSI_high`.
* **Execution model**: next-day approximation (`position.shift(1)`).
* **Backtest**: vectorized; equity = cumulative product of (1 + returns).
* **Risk-free**: optional annual rate converted to daily for Sharpe.

---

## Optional: Alpaca Paper Trading 🧪

Prepare environment variables (you’ll wire the “Send to Paper” button later):

```bash
export APCA_API_KEY_ID="YOUR_KEY"
export APCA_API_SECRET_KEY="YOUR_SECRET"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
```

---

## Roadmap 🗺️

* ML baseline: Gradient Boosting + **TimeSeriesSplit** (walk-forward)
* Strategy comparison: SMA+RSI vs ML classifier
* **Alpaca** integration: one-click paper trade, position sizing, risk caps
* Risk controls: volatility targeting, ATR stops
* Persistence: save/load experiments (`.yaml`)
* Tests: unit tests for RSI, drawdown, Sharpe

---

## Troubleshooting 🧯

**“Data must be 1-dimensional, got ndarray of shape (n, 1)”**

* Use a **Series** (1D) instead of a single-column DataFrame (2D).
* Prefer `df['Close']` over `df[['Close']]`.
* This repo includes safeguards (`.squeeze()` + normalization in `utils.py`).

**No data returned**

* Check ticker and date range. Some symbols lack history or changed names.

**Multiple symbols**

* The MVP handles **one symbol at a time**. Multi-asset baskets need an additional layer.

---

## Contributing 🤝

PRs are welcome! Please include:

* A clear description of the change
* A minimal reproducible example
* Tests for any new utility functions

---

## License 📄

MIT © Your Name

---

## Important Disclaimer ⚠️

This project is for **research and education** only. It is **not investment advice**.
Past performance (including **backtests**) is **not** indicative of future results.
If you plan to sell signals or personalized guidance, consult applicable regulations and legal counsel.
