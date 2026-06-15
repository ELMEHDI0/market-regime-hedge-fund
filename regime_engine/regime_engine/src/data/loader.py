"""
Data Layer
==========
Responsibilities:
  1. Pull daily OHLCV for the configured tickers (yfinance).
  2. Clean: drop bad rows, forward-fill *small* gaps, kill timezone, align
     every asset onto one shared trading-day calendar.
  3. Store as Parquet (one file per asset) or SQLite.

If yfinance is unavailable or the network is blocked (e.g. sandbox), the
loader transparently falls back to a *regime-switching synthetic* generator
so the full pipeline still runs and can be validated end-to-end. The data
source actually used is recorded in outputs/market_data/_SOURCE.txt.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def load_market_data(cfg: dict) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame[open,high,low,close,volume]} on a shared index."""
    d = cfg["data"]
    source = d.get("source", "auto")
    raw: dict[str, pd.DataFrame] = {}
    used = "synthetic"

    if source in ("auto", "yfinance"):
        try:
            raw = _fetch_yfinance(d["tickers"], d["start"], d["end"])
            if raw:
                used = "yfinance"
        except Exception as e:  # noqa: BLE001
            if source == "yfinance":
                raise
            print(f"[data] yfinance unavailable ({e}); using synthetic data.")

    if not raw:
        raw = _synthetic(d["tickers"], d["start"], d["end"])
        used = "synthetic"

    clean = _clean_and_align(raw, calendar=d.get("align_calendar", d["tickers"][0]))
    _store(clean, d, used)
    print(f"[data] source={used}  assets={list(clean)}  "
          f"rows={len(next(iter(clean.values())))}")
    return clean


# --------------------------------------------------------------------------
# yfinance fetch
# --------------------------------------------------------------------------
def _fetch_yfinance(tickers, start, end) -> dict[str, pd.DataFrame]:
    import yfinance as yf  # imported lazily so the package isn't required offline

    out = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True,
                         progress=False, threads=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        out[t] = df
    return out


# --------------------------------------------------------------------------
# Cleaning + alignment
# --------------------------------------------------------------------------
def _clean_and_align(raw: dict[str, pd.DataFrame], calendar: str) -> dict[str, pd.DataFrame]:
    cleaned = {}
    for t, df in raw.items():
        df = df.copy()
        # strip timezone -> pure trading dates
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # non-positive prices are corrupt
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        df["volume"] = df["volume"].fillna(0)
        cleaned[t] = df

    # Build a master calendar from the chosen anchor (fallback: union).
    if calendar in cleaned:
        master = cleaned[calendar].index
    else:
        master = sorted(set().union(*[d.index for d in cleaned.values()]))
        master = pd.DatetimeIndex(master)

    aligned = {}
    for t, df in cleaned.items():
        df = df.reindex(master)
        # Forward-fill only *short* gaps (<=3 days) so we don't fabricate weeks
        # of data when an asset (e.g. crypto vs equities) simply isn't trading.
        df[["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].ffill(limit=3)
        )
        df["volume"] = df["volume"].fillna(0)
        df = df.dropna(subset=["close"])
        aligned[t] = df.reindex(master).dropna(subset=["close"])
    return aligned


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def _store(data: dict[str, pd.DataFrame], dcfg: dict, source: str) -> None:
    path = Path(dcfg["store_path"])
    path.mkdir(parents=True, exist_ok=True)
    (path / "_SOURCE.txt").write_text(source)

    if dcfg.get("storage", "parquet") == "sqlite":
        con = sqlite3.connect(path / "market.db")
        for t, df in data.items():
            df.reset_index(names="date").to_sql(
                _safe(t), con, if_exists="replace", index=False)
        con.close()
    else:
        for t, df in data.items():
            df.to_parquet(path / f"{_safe(t)}.parquet")


def _safe(t: str) -> str:
    return t.replace("-", "_").replace("=", "_").replace(".", "_")


# --------------------------------------------------------------------------
# Synthetic, regime-switching data (for offline / CI runs)
# --------------------------------------------------------------------------
def _synthetic(tickers, start, end) -> dict[str, pd.DataFrame]:
    """
    Generates daily OHLCV whose drift/vol switch between four hidden regimes
    (bull, bear, range, crash). This is NOT a market model — it exists only so
    the regime detector and strategies have realistic structure to chew on
    when live data can't be fetched.
    """
    rng = np.random.default_rng(7)
    end = end or pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)

    # Shared hidden regime path (assets co-move but with different betas).
    states = _regime_path(n, rng)
    # per-regime (daily drift, daily vol)
    params = {
        0: (0.0011, 0.006),   # bull / trend up   (cleaner, exploitable trend)
        1: (-0.0013, 0.009),  # bear / trend down
        2: (0.0000, 0.005),   # range
        3: (-0.0030, 0.028),  # crash
    }

    out = {}
    betas = {"SPY": 1.0, "QQQ": 1.3, "GLD": 0.3, "BTC-USD": 2.2, "EURUSD": 0.25}
    base = {"SPY": 200, "QQQ": 180, "GLD": 110, "BTC-USD": 8000, "EURUSD": 1.10}
    for t in tickers:
        beta = betas.get(t, 1.0)
        idio = rng.normal(0, 1, n)
        rets = np.empty(n)
        for i, s in enumerate(states):
            mu, sig = params[s]
            # market component scaled by beta + idiosyncratic noise
            rets[i] = beta * (mu + sig * rng.normal()) + 0.4 * sig * idio[i]
        price = base.get(t, 100) * np.exp(np.cumsum(rets))
        close = pd.Series(price, index=idx)
        # build OHLC around close
        intraday = np.abs(rng.normal(0, 0.004, n)) + 0.001
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, n))
        vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
        out[t] = pd.DataFrame(
            {"open": open_.values, "high": high.values, "low": low.values,
             "close": close.values, "volume": vol}, index=idx)
    return out


def _regime_path(n: int, rng) -> np.ndarray:
    """Persistent 4-state Markov chain so regimes last weeks, not days."""
    P = np.array([
        [0.990, 0.003, 0.005, 0.002],   # bull is sticky and dominant
        [0.020, 0.960, 0.015, 0.005],   # bear is transient
        [0.030, 0.010, 0.955, 0.005],   # range
        [0.100, 0.050, 0.050, 0.800],   # crash mean-reverts back to bull
    ])
    s = np.empty(n, dtype=int)
    s[0] = 0
    for i in range(1, n):
        s[i] = rng.choice(4, p=P[s[i - 1]])
    return s
