"""
Causal technical indicators.

Every function here uses only data up to and including the current bar.
Nothing peeks into the future, so feeding the output into a backtest and
shifting by one bar is sufficient to guarantee no lookahead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index — trend *strength* (0-100), direction-agnostic."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def realised_vol(close: pd.Series, window: int = 20, periods: int = 252) -> pd.Series:
    """Annualised rolling realised volatility from log returns."""
    r = np.log(close / close.shift(1))
    return r.rolling(window).std() * np.sqrt(periods)


def rolling_percentile(s: pd.Series, window: int = 252) -> pd.Series:
    """Rank of the latest value within its trailing window, in [0, 1]. Causal."""
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: (x[-1] >= x).mean(), raw=True)


def rolling_drawdown(close: pd.Series, window: int = 21) -> pd.Series:
    """Drawdown vs the rolling max over `window` days (a fast crash detector)."""
    roll_max = close.rolling(window, min_periods=1).max()
    return close / roll_max - 1.0
