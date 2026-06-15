"""
Strategy Layer
==============
Three strategies, each GATED to specific regimes. A strategy that is gated
off contributes a flat (0) target and says *why* it is inactive — satisfying
the "explain why each strategy is active or inactive" requirement.

Each strategy is a small state machine producing, per bar:
    target_dir : desired position direction in {-1, 0, +1}  (pre-sizing)
    stop_dist  : price distance to the protective stop (for risk sizing)
    note       : human-readable reason for the current state

Sizing (units/notional) is delegated to the RiskEngine downstream.
Signals are computed on bar t using only info up to t; the backtester
executes them on t+1, so there is no lookahead and no repainting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..regime import indicators as ind
from ..regime.detectors import TREND_UP, TREND_DOWN, RANGE, CRASH


class BaseStrategy:
    name = "base"
    active_regimes: set[str] = set()

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _atr(self, df):
        return ind.atr(df, self.cfg["regime"]["atr_window"])

    def generate(self, df: pd.DataFrame, feats: pd.DataFrame,
                 regimes: pd.Series) -> pd.DataFrame:
        raise NotImplementedError

    def gate_note(self, regime: str) -> str:
        if regime in self.active_regimes:
            return f"ACTIVE (regime {regime} ∈ {sorted(self.active_regimes)})"
        return f"INACTIVE (regime {regime} ∉ {sorted(self.active_regimes)})"


# ---------------------------------------------------------------------
# 1. Trend following — only in TREND_UP / TREND_DOWN
# ---------------------------------------------------------------------
class TrendFollow(BaseStrategy):
    name = "trend_follow"
    active_regimes = {TREND_UP, TREND_DOWN}

    def generate(self, df, feats, regimes):
        p = self.cfg["strategy"]["trend"]
        atr = self._atr(df)
        ma = feats["ma_fast"]
        close = df["close"]

        out = _blank(df.index)
        pos = 0
        for dt in df.index:
            reg = regimes.get(dt)
            a = atr.get(dt, np.nan)
            c = close.get(dt, np.nan)
            m = ma.get(dt, np.nan)
            note = self.gate_note(reg)

            if reg not in self.active_regimes or np.isnan(a) or np.isnan(m):
                pos = 0
                out.loc[dt] = [0, np.nan, note + " -> flat"]
                continue

            if reg == TREND_UP:
                # enter/stay long while price holds above fast MA
                if c > m - p["pullback_atr"] * a:
                    pos = 1
                    note += f"; long: close>{p['pullback_atr']}·ATR below fastMA"
                elif c < m - p["trail_atr"] * a:
                    pos = 0
                    note += "; exit long: broke trailing ATR band"
            elif reg == TREND_DOWN:
                if c < m + p["pullback_atr"] * a:
                    pos = -1
                    note += f"; short: close<{p['pullback_atr']}·ATR above fastMA"
                elif c > m + p["trail_atr"] * a:
                    pos = 0
                    note += "; exit short: broke trailing ATR band"

            out.loc[dt] = [pos, p["stop_atr"] * a, note]
        return out


# ---------------------------------------------------------------------
# 2. Mean reversion — only in RANGE
# ---------------------------------------------------------------------
class MeanReversion(BaseStrategy):
    name = "mean_reversion"
    active_regimes = {RANGE}

    def generate(self, df, feats, regimes):
        p = self.cfg["strategy"]["mean_reversion"]
        atr = self._atr(df)
        close = df["close"]
        ma = close.rolling(p["z_window"]).mean()
        sd = close.rolling(p["z_window"]).std()
        z = (close - ma) / sd

        out = _blank(df.index)
        pos = 0
        for dt in df.index:
            reg = regimes.get(dt)
            note = self.gate_note(reg)
            zz = z.get(dt, np.nan)
            a = atr.get(dt, np.nan)
            if reg not in self.active_regimes or np.isnan(zz) or np.isnan(a):
                pos = 0
                out.loc[dt] = [0, np.nan, note + " -> flat"]
                continue

            if pos == 0:
                if zz <= -p["z_entry"]:
                    pos = 1; note += f"; long fade: z={zz:.2f} ≤ -{p['z_entry']}"
                elif zz >= p["z_entry"]:
                    pos = -1; note += f"; short fade: z={zz:.2f} ≥ {p['z_entry']}"
                else:
                    note += f"; wait: |z|={abs(zz):.2f} < {p['z_entry']}"
            else:
                if abs(zz) <= p["z_exit"]:
                    note += f"; exit: reverted to mean (z={zz:.2f})"
                    pos = 0
                else:
                    note += f"; hold fade (z={zz:.2f})"

            out.loc[dt] = [pos, p["stop_atr"] * a, note]
        return out


# ---------------------------------------------------------------------
# 3. Volatility breakout — only in HIGH_VOL_CRASH
# ---------------------------------------------------------------------
class VolBreakout(BaseStrategy):
    name = "vol_breakout"
    active_regimes = {CRASH}

    def generate(self, df, feats, regimes):
        p = self.cfg["strategy"]["vol_breakout"]
        atr = self._atr(df)
        # Donchian channel from PRIOR bars only (shift 1) -> no lookahead.
        hi = df["high"].rolling(p["donchian"]).max().shift(1)
        lo = df["low"].rolling(p["donchian"]).min().shift(1)
        close = df["close"]

        out = _blank(df.index)
        pos = 0
        for dt in df.index:
            reg = regimes.get(dt)
            note = self.gate_note(reg)
            a = atr.get(dt, np.nan)
            c = close.get(dt, np.nan)
            h = hi.get(dt, np.nan); l = lo.get(dt, np.nan)
            if reg not in self.active_regimes or np.isnan(a) or np.isnan(h):
                pos = 0
                out.loc[dt] = [0, np.nan, note + " -> flat"]
                continue

            if c > h:
                pos = 1; note += f"; breakout up > {p['donchian']}d high"
            elif c < l:
                pos = -1; note += f"; breakdown < {p['donchian']}d low"
            else:
                note += "; inside channel, hold"
            out.loc[dt] = [pos, p["stop_atr"] * a, note]
        return out


def _blank(index) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=["target_dir", "stop_dist", "note"])


def all_strategies(cfg: dict) -> list[BaseStrategy]:
    return [TrendFollow(cfg), MeanReversion(cfg), VolBreakout(cfg)]
