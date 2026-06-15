"""
Risk Engine
===========
Turns raw strategy intents (direction + stop distance) into actual portfolio
weights under hard constraints.

Controls:
  * Risk-per-trade budgeting (0.5%-1%): size so a stop-out costs ~risk_per_trade.
  * Volatility-targeted sizing: scale toward a constant portfolio vol.
  * Correlation reduction: haircut weights of assets that move together.
  * Exposure caps: per-asset weight cap + total gross cap.
  * Portfolio drawdown circuit breaker (hard stop) with optional re-arm.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class RiskEngine:
    def __init__(self, cfg: dict):
        r = cfg["risk"]
        self.risk_per_trade = r["risk_per_trade"]
        self.max_dd = r["max_portfolio_drawdown"]
        self.cooldown = int(r.get("drawdown_cooldown", 21))   # 0 => permanent halt
        self.vol_target = r["vol_target_annual"]
        self.corr_window = r["corr_window"]
        self.corr_threshold = r["corr_threshold"]
        self.max_gross = r["max_gross_exposure"]
        self.max_w = r["max_weight_per_asset"]
        self.periods = cfg["report"]["periods_per_year"]
        # breaker state
        self._state = "ARMED"        # ARMED | HALTED | GRACE
        self._counter = 0
        self._hwm = None             # all-time high-water mark (never reset)
        self.n_breaches = 0

    # ------------------------------------------------------------------
    def update_drawdown(self, latest_equity: float) -> bool:
        """
        Drawdown circuit breaker on drawdown from the all-time high-water mark
        (HWM is never reset -> no 'death spiral' of stacked 20% losses).

          ARMED  : trading. Trip to HALTED if DD-from-HWM <= -max_dd.
          HALTED : forced flat for `cooldown` days, then -> GRACE.
          GRACE  : trading resumes for `cooldown` days and will NOT re-trip on
                   the same drawdown, giving the book room to recover; a fresh
                   catastrophe (DD <= -2*max_dd) still forces an immediate HALT.

        cooldown == 0 => permanent hard stop (never leaves HALTED).
        Returns True only while forced flat (HALTED).
        """
        if self._hwm is None:
            self._hwm = latest_equity
        self._hwm = max(self._hwm, latest_equity)
        dd = latest_equity / self._hwm - 1.0

        if self._state == "HALTED":
            if self.cooldown <= 0:
                return True                       # permanent stop
            self._counter -= 1
            if self._counter <= 0:
                self._state, self._counter = "GRACE", self.cooldown
            return True

        if self._state == "GRACE":
            self._counter -= 1
            if dd <= -2.0 * self.max_dd:          # catastrophe override
                self._state, self._counter = "HALTED", self.cooldown
                self.n_breaches += 1
                return True
            if self._counter <= 0:
                self._state = "ARMED"
            return False

        # ARMED
        if dd <= -self.max_dd:
            self._state, self._counter = "HALTED", self.cooldown
            self.n_breaches += 1
            return True
        return False

    @property
    def halted(self) -> bool:
        return self._state == "HALTED"

    # ------------------------------------------------------------------
    def size(self, dirs: dict[str, float], stops: dict[str, float],
             prices: dict[str, float], ret_window: pd.DataFrame) -> dict[str, float]:
        """Target portfolio weights per asset for one day (uses only past data)."""
        if self.halted:
            return {a: 0.0 for a in dirs}

        weights = {}
        for a, d in dirs.items():
            s = stops.get(a, np.nan)
            if d == 0 or np.isnan(s) or s <= 0:
                weights[a] = 0.0
                continue
            stop_pct = max(s / prices[a], 1e-4)
            weights[a] = d * (self.risk_per_trade / stop_pct)

        weights = self._vol_target(weights, ret_window)
        weights = self._corr_haircut(weights, ret_window)
        for a in weights:
            weights[a] = float(np.clip(weights[a], -self.max_w, self.max_w))
        gross = sum(abs(w) for w in weights.values())
        if gross > self.max_gross and gross > 0:
            scale = self.max_gross / gross
            weights = {a: w * scale for a, w in weights.items()}
        return weights

    # ------------------------------------------------------------------
    def _vol_target(self, weights, ret_window) -> dict:
        if ret_window is None or len(ret_window) < 20:
            return weights
        w = pd.Series(weights).reindex(ret_window.columns).fillna(0.0)
        cov = ret_window.cov() * self.periods
        port_vol = np.sqrt(max(float(w.values @ cov.values @ w.values), 1e-12))
        if port_vol <= 1e-8:
            return weights
        scale = float(np.clip(self.vol_target / port_vol, 0.0, 3.0))
        return {a: weights[a] * scale for a in weights}

    def _corr_haircut(self, weights, ret_window) -> dict:
        if ret_window is None or len(ret_window) < self.corr_window // 2:
            return weights
        corr = ret_window.corr()
        active = [a for a, w in weights.items() if abs(w) > 1e-9]
        out = dict(weights)
        for a in active:
            excess = 0.0
            for b in active:
                if b == a:
                    continue
                c = corr.loc[a, b] if a in corr.index and b in corr.columns else 0.0
                if np.sign(weights[a]) == np.sign(weights[b]) and abs(c) >= self.corr_threshold:
                    excess += abs(c) - self.corr_threshold
            out[a] = weights[a] / (1.0 + excess)
        return out
