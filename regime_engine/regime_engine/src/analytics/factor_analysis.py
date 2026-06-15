"""
Factor Exposure Analysis  (Priority #2)
======================================
Regress the strategy's daily returns on a set of style factors to decompose
performance into ALPHA + factor BETAS, and report institutional risk-adjusted
stats: Information Ratio and Tracking Error vs the benchmark.

Factors are constructed from the traded universe (no external data needed):
  * MKT  : benchmark (SPY) excess return  -> market beta
  * MOM  : cross-sectional 12-1 momentum  (long winners / short losers)
  * LOWVOL: low-vol minus high-vol         (defensive factor)
  * CARRY: trend of the FX/gold sleeve     (rough carry/safe-haven proxy)

SIZE and VALUE are flagged N/A unless a small-cap / value series is supplied;
the regression interface accepts an external factor DataFrame to plug in real
Fama-French / AQR factors later.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_factors(data: dict[str, pd.DataFrame], benchmark: str,
                  rf_daily: float) -> pd.DataFrame:
    rets = pd.DataFrame({a: data[a]["close"].pct_change() for a in data})
    idx = rets.index

    mkt = rets[benchmark] - rf_daily

    # 12-1 momentum: rank assets by trailing 252d ret skipping last 21d
    trail = (data_close(data).shift(21) / data_close(data).shift(252) - 1)
    mom = _long_short(rets, trail)

    # low-vol minus high-vol (rank by trailing 63d vol)
    vol63 = rets.rolling(63).std()
    lowvol = _long_short(rets, -vol63)             # long low vol => negate rank key

    # carry / safe-haven proxy: trend of GLD & EURUSD sleeve if present
    haven_assets = [a for a in ("GLD", "EURUSD") if a in rets.columns]
    carry = rets[haven_assets].mean(axis=1) if haven_assets else pd.Series(0.0, index=idx)

    f = pd.DataFrame({"MKT": mkt, "MOM": mom, "LOWVOL": lowvol, "CARRY": carry})
    return f


def data_close(data):
    return pd.DataFrame({a: data[a]["close"] for a in data})


def _long_short(rets: pd.DataFrame, signal: pd.DataFrame) -> pd.Series:
    """Daily return of a dollar-neutral long-top / short-bottom basket."""
    out = pd.Series(0.0, index=rets.index)
    sig = signal.reindex(columns=rets.columns)
    for i in range(len(rets)):
        row = sig.iloc[i].dropna()
        if len(row) < 2:
            continue
        hi = row.nlargest(max(1, len(row) // 2)).index
        lo = row.nsmallest(max(1, len(row) // 2)).index
        out.iloc[i] = rets[hi].iloc[i].mean() - rets[lo].iloc[i].mean()
    return out


def factor_regression(strategy_ret: pd.Series, factors: pd.DataFrame,
                      benchmark_ret: pd.Series, rf_daily: float,
                      periods: int = 252) -> dict:
    df = pd.concat([strategy_ret.rename("y"), factors], axis=1).dropna()
    if len(df) < 60:
        return {"error": "insufficient overlap for regression"}
    y = (df["y"] - rf_daily).values
    X = df[factors.columns].values
    X = np.column_stack([np.ones(len(X)), X])              # intercept = alpha

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    alpha_daily = beta[0]
    exposures = dict(zip(factors.columns, np.round(beta[1:], 3)))

    # Information ratio & tracking error vs benchmark
    aligned = pd.concat([strategy_ret, benchmark_ret], axis=1).dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std() * np.sqrt(periods)
    ir = (active.mean() * periods) / te if te > 0 else 0.0

    return {
        "alpha_annual": round(float(alpha_daily * periods), 4),
        "market_beta": exposures.get("MKT", 0.0),
        "factor_betas": exposures,
        "r_squared": round(float(r2), 3),
        "information_ratio": round(float(ir), 3),
        "tracking_error_annual": round(float(te), 4),
        "note": "SIZE/VALUE = N/A (supply external factor columns to extend).",
    }


def run_factor_analysis(results: dict, data: dict, cfg: dict) -> dict:
    rf_daily = cfg["report"]["rf_annual"] / cfg["report"]["periods_per_year"]
    strat_ret = results["equity"].pct_change()
    bench_ret = data[cfg["data"]["benchmark"]]["close"].pct_change()
    factors = build_factors(data, cfg["data"]["benchmark"], rf_daily)
    return factor_regression(strat_ret, factors, bench_ret, rf_daily,
                             cfg["report"]["periods_per_year"])
