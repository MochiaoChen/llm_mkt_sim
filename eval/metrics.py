# file: eval/metrics.py
from __future__ import annotations
"""
Basic evaluation metrics for market quality and stylized facts.

Inputs
------
- trades_df: columns ["time","price","qty","buy_agent","sell_agent","trade_id"]
- accounts_df: ["time","agent","cash","inventory","equity","fees_paid"]
- snaps_df: ["time","mid","spread","best_bid","best_ask","bid_depth","ask_depth"]

Outputs
-------
- compute_basic_metrics(...) -> dict with key market-quality and stylized metrics.
- helper functions for returns, kurtosis, ACF(1), variance ratio.

Notes
-----
This is intentionally lightweight for MVP. For paper-grade analysis,
you'll likely extend with Kyle's lambda, Hasbrouck info shares, and
execution slippage metrics per agent/trade.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _nanstd(x: pd.Series) -> float:
    return float(np.nanstd(x.values))


def _kurtosis(x: pd.Series) -> float:
    # Fisher definition (excess kurtosis): scipy.stats.kurtosis(..., fisher=True, bias=False)
    x = x.dropna()
    if len(x) < 4:
        return float("nan")
    m = x.mean()
    s2 = ((x - m) ** 2).mean()
    if s2 <= 0:
        return float("nan")
    m4 = ((x - m) ** 4).mean()
    kurt = m4 / (s2 ** 2) - 3.0
    return float(kurt)


def _acf1(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return float("nan")
    x0 = x[:-1]
    x1 = x[1:]
    num = ((x1 - x1.mean()) * (x0 - x0.mean())).sum()
    den = ((x - x.mean()) ** 2).sum()
    if den == 0:
        return float("nan")
    return float(num / den)


def _variance_ratio(x: pd.Series, q: int = 2) -> float:
    """
    Lo-MacKinlay variance ratio (simplified, no heteroskedasticity correction).
    VR(q) ~ Var(r_agg) / (q * Var(r_1))
    """
    x = x.dropna()
    if len(x) <= q:
        return float("nan")
    r1 = x.diff()
    rq = x.diff(q)
    v1 = r1.var()
    vq = rq.var()
    if v1 is None or vq is None or v1 <= 0:
        return float("nan")
    return float(vq / (q * v1))


def _to_returns_from_mid(snaps_df: pd.DataFrame) -> pd.Series:
    # Use mid if available; fallback to best of bid/ask, else NaN
    px = snaps_df["mid"].copy()
    if px.isna().all():
        # fallback: mean of best bid/ask
        approx = (snaps_df["best_bid"].fillna(method="ffill") + snaps_df["best_ask"].fillna(method="ffill")) / 2.0
        px = approx
    r = px.ffill().bfill().dropna().pct_change()
    return r


def _pnl_summary(accounts_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Return (avg_equity_end, sharpe_like, max_drawdown) across agents.
    """
    if accounts_df.empty:
        return float("nan"), float("nan"), float("nan")
    # Pivot equity time-series per agent
    piv = accounts_df.pivot_table(index="time", columns="agent", values="equity")
    # Compute per-agent return series (diff in equity)
    rets = piv.diff().fillna(0.0)
    # "Sharpe-like": mean/std at per-tick equity change (no risk-free, no annualization)
    mu = rets.mean()
    sd = rets.std().replace(0.0, np.nan)
    sharpe_like = float(np.nanmean(mu / sd))

    # Max drawdown per agent based on equity curve
    def _max_dd(series: pd.Series) -> float:
        roll_max = series.cummax()
        dd = (series - roll_max) / roll_max.replace(0.0, np.nan)
        return float(dd.min())

    mdds = piv.apply(_max_dd)
    max_drawdown = float(mdds.mean())

    # Equity end
    equity_end = float(piv.iloc[-1].mean())
    return equity_end, sharpe_like, max_drawdown


def compute_basic_metrics(
    trades_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    snaps_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute a compact set of metrics to quickly assess runs.

    Returns
    -------
    dict with keys:
      - mean_spread, median_spread
      - avg_bid_depth, avg_ask_depth
      - ret_std, ret_kurtosis, ret_acf1, var_ratio_q2
      - equity_end_avg, sharpe_like, max_drawdown_avg
      - trade_count, turnover_est
    """
    out: Dict[str, float] = {}

    # Market quality (spread/depth)
    if not snaps_df.empty:
        out["mean_spread"] = float(snaps_df["spread"].dropna().mean())
        out["median_spread"] = float(snaps_df["spread"].dropna().median())
        out["avg_bid_depth"] = float(snaps_df["bid_depth"].dropna().mean())
        out["avg_ask_depth"] = float(snaps_df["ask_depth"].dropna().mean())
    else:
        out["mean_spread"] = out["median_spread"] = float("nan")
        out["avg_bid_depth"] = out["avg_ask_depth"] = float("nan")

    # Stylized facts on returns
    rets = _to_returns_from_mid(snaps_df)
    out["ret_std"] = float(rets.std()) if len(rets) > 1 else float("nan")
    out["ret_kurtosis"] = _kurtosis(rets)
    out["ret_acf1"] = _acf1(rets)
    out["var_ratio_q2"] = _variance_ratio(snaps_df["mid"].ffill(), q=2) if "mid" in snaps_df else float("nan")

    # PnL / account health
    eq_end, sharpe_like, mdd = _pnl_summary(accounts_df)
    out["equity_end_avg"] = eq_end
    out["sharpe_like"] = sharpe_like
    out["max_drawdown_avg"] = mdd

    # Trading activity
    out["trade_count"] = int(len(trades_df))
    out["turnover_est"] = float(trades_df["qty"].sum()) if "qty" in trades_df else float("nan")

    return out
