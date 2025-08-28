import pandas as pd
import numpy as np

def _to_series(x, name=None):
    # Asegura 1D (Series). Si viene (n,1) DataFrame/ndarray -> squeeze.
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise ValueError(f"Expected 1D series, got DataFrame with {x.shape[1]} columns")
    elif isinstance(x, np.ndarray):
        x = np.squeeze(x)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got array with shape {x.shape}")
        x = pd.Series(x)
    # Ahora x debería ser Serie
    x = x.copy()
    if name is not None:
        x.name = name
    return x

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    series = _to_series(series, name="close")
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(length).mean()
    roll_down = pd.Series(down, index=series.index).rolling(length).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def vector_backtest(close, position, rf: float = 0.0) -> pd.DataFrame:
    prices = _to_series(close, name="close").astype(float)
    pos = _to_series(position, name="position").reindex(prices.index).astype(float)
    pos = pos.shift(1).fillna(0.0)  # ejecutar al siguiente día

    rets = prices.pct_change().fillna(0.0)
    strat_rets = pos * rets

    # rf anual -> diario
    daily_rf = (1.0 + rf) ** (1/252) - 1.0
    rf_series = pd.Series(daily_rf, index=prices.index, name="rf")

    excess = strat_rets - rf_series
    equity = (1.0 + strat_rets).cumprod()

    out = pd.DataFrame({
        "returns": strat_rets,
        "excess": excess,
        "rf_series": rf_series,
        "equity_curve": equity
    }, index=prices.index)
    return out

def metrics_report(equity_curve, rets, rf_series) -> dict:
    equity_curve = _to_series(equity_curve, name="equity")
    rets = _to_series(rets, name="returns")
    rf_series = _to_series(rf_series, name="rf")

    total_return = equity_curve.iloc[-1] - 1.0
    n_days = len(rets)
    if n_days <= 1:
        cagr = 0.0
    else:
        years = n_days / 252.0
        cagr = equity_curve.iloc[-1] ** (1/years) - 1.0

    rolling_max = equity_curve.cummax()
    dd = equity_curve / rolling_max - 1.0
    max_dd = dd.min()

    sharpe = 0.0
    denom = rets.sub(rf_series, fill_value=0).std(ddof=0)
    if denom > 0:
        sharpe = np.sqrt(252) * (rets.sub(rf_series, fill_value=0).mean() / denom)

    return {
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_dd_pct": float(max_dd * 100.0),
        "sharpe": float(sharpe)
    }
