"""
Portfolio Construction Layer  (Priority #1)
==========================================
Allocation schemes operating on a returns/cov estimate:

  * equal_weight
  * inverse_vol          (a.k.a. naive risk parity)
  * risk_parity          (equal risk contribution, iterative)
  * vol_target           (scale any weight vector to a target vol)
  * max_diversification   (maximise the diversification ratio)

Plus REGIME-DEPENDENT strategic allocations: a different target mix per
market regime (bull / bear / range / crisis), e.g.

    BULL    -> 60% SPY, 20% QQQ, 10% GLD, 10% BTC
    CRISIS  -> 50% GLD, 30% CASH, 20% (bond proxy)

A 'CASH' sleeve (zero return) is synthesised since the data universe has no
risk-free instrument; bonds are proxied by GLD when absent (documented).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..regime.detectors import TREND_UP, TREND_DOWN, RANGE, CRASH


# ---------------------------------------------------------------------
# Static allocation schemes
# ---------------------------------------------------------------------
def equal_weight(assets: list[str]) -> dict:
    w = 1.0 / len(assets)
    return {a: w for a in assets}


def inverse_vol(cov: pd.DataFrame) -> dict:
    vol = np.sqrt(np.diag(cov.values))
    inv = 1.0 / np.where(vol > 0, vol, np.inf)
    w = inv / inv.sum()
    return dict(zip(cov.columns, w))


def risk_parity(cov: pd.DataFrame, iters: int = 500) -> dict:
    """Equal-risk-contribution weights via fixed-point iteration."""
    Sigma = cov.values
    n = Sigma.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = Sigma @ w                      # marginal risk contribution
        rc = w * mrc                         # risk contribution
        target = rc.mean()
        w *= (target / np.where(rc > 1e-12, rc, 1e-12)) ** 0.5
        w = np.clip(w, 0, None)
        w /= w.sum()
    return dict(zip(cov.columns, w))


def max_diversification(cov: pd.DataFrame) -> dict:
    """Maximise diversification ratio = (wᵀσ) / sqrt(wᵀΣw), long-only, sum=1."""
    Sigma = cov.values
    sig = np.sqrt(np.diag(Sigma))
    n = len(sig)

    def neg_dr(w):
        num = w @ sig
        den = np.sqrt(max(w @ Sigma @ w, 1e-12))
        return -num / den

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(neg_dr, np.ones(n) / n, bounds=bnds, constraints=cons,
                   method="SLSQP")
    w = res.x if res.success else np.ones(n) / n
    return dict(zip(cov.columns, w))


def scale_to_vol(weights: dict, cov: pd.DataFrame, target: float, periods: int = 252) -> dict:
    w = pd.Series(weights).reindex(cov.columns).fillna(0.0).values
    vol = np.sqrt(max(w @ cov.values @ w, 1e-12)) * np.sqrt(periods)
    k = float(np.clip(target / vol, 0.0, 3.0)) if vol > 0 else 1.0
    return {a: weights[a] * k for a in weights}


# ---------------------------------------------------------------------
# Regime-dependent strategic allocations
# ---------------------------------------------------------------------
REGIME_PRESETS = {
    TREND_UP:   {"SPY": 0.60, "QQQ": 0.20, "GLD": 0.10, "BTC-USD": 0.10},
    RANGE:      {"SPY": 0.30, "QQQ": 0.10, "GLD": 0.30, "EURUSD": 0.10, "CASH": 0.20},
    TREND_DOWN: {"GLD": 0.40, "EURUSD": 0.20, "CASH": 0.40},
    CRASH:      {"GLD": 0.50, "CASH": 0.30, "BONDS": 0.20},
}


def regime_allocation(regime: str, assets: list[str]) -> dict:
    """Resolve a preset against the available universe (proxy missing sleeves)."""
    preset = REGIME_PRESETS.get(regime, {}).copy()
    out = {a: 0.0 for a in assets + ["CASH"]}
    for sleeve, w in preset.items():
        if sleeve == "BONDS":                       # bond proxy: GLD if no bond asset
            sleeve = "GLD" if "GLD" in assets else "CASH"
        if sleeve in out:
            out[sleeve] += w
        else:
            out["CASH"] += w                        # unavailable -> park in cash
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


# ---------------------------------------------------------------------
# Build + lightly backtest a regime-driven strategic portfolio
# ---------------------------------------------------------------------
def regime_portfolio_backtest(data: dict[str, pd.DataFrame], market_regime: pd.Series,
                              cfg: dict) -> dict:
    """
    Strategic (long-only) overlay: each day hold the preset mix for the current
    MARKET regime (taken from the benchmark asset's regime). CASH earns 0.
    Returns its equity curve + the time series of target weights.
    """
    assets = list(data)
    rets = pd.DataFrame({a: data[a]["close"].pct_change() for a in assets})
    rets["CASH"] = 0.0
    idx = rets.index

    weights = pd.DataFrame(0.0, index=idx, columns=list(rets.columns))
    reg = market_regime.reindex(idx).ffill()
    for dt in idx:
        alloc = regime_allocation(reg.get(dt, RANGE), assets)
        for k, v in alloc.items():
            if k in weights.columns:
                weights.loc[dt, k] = v

    # lag weights by 1 day (decide at close t, hold t+1) — no lookahead
    port_ret = (weights.shift(1) * rets).sum(axis=1).fillna(0.0)
    equity = cfg["backtest"]["initial_capital"] * (1 + port_ret).cumprod()
    equity.name = "regime_portfolio_equity"
    return {"equity": equity, "weights": weights}


def all_static_allocations(cov: pd.DataFrame, cfg: dict) -> dict:
    """Convenience: compute every static scheme on one covariance estimate."""
    assets = list(cov.columns)
    target = cfg["risk"]["vol_target_annual"]
    ppy = cfg["report"]["periods_per_year"]
    ew = equal_weight(assets)
    return {
        "equal_weight": ew,
        "inverse_vol": inverse_vol(cov),
        "risk_parity": risk_parity(cov),
        "max_diversification": max_diversification(cov),
        "vol_targeted_equal_weight": scale_to_vol(ew, cov, target, ppy),
    }
