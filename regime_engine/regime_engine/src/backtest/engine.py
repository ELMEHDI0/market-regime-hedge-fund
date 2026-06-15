"""
Backtesting Engine
==================
Portfolio-level, daily, path-dependent loop.

No-lookahead guarantees:
  * All signals/features on bar t use data up to and including close(t).
  * Weights decided at close(t) are only applied to the return from t -> t+1.
  * Donchian/percentile/HMM-walkforward channels are shifted so a bar never
    sees its own future.

Costs:
  * Commission: commission_bps per side, charged on |Δweight| (turnover).
  * Slippage: fixed bps OR volatility-based (slippage_atr_mult · ATR%).

Outputs (returned + written to disk by the report layer):
  equity curve, drawdown curve, daily weights, regimes, full trade list.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..regime.detectors import RegimeEnsemble
from ..regime import indicators as ind
from ..strategies.strategies import all_strategies
from ..risk.risk_engine import RiskEngine


class Backtester:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bt = cfg["backtest"]

    # ------------------------------------------------------------------
    def _per_asset_signals(self, data: dict[str, pd.DataFrame]):
        """Compute regime + merged strategy intent for every asset."""
        signals = {}
        regimes_out = {}
        ens_store = {}
        for t, df in data.items():
            ens = RegimeEnsemble(self.cfg)
            reg = ens.run(df)
            feats = ens.features
            regime_s = reg["regime"]

            strat_targets = [s.generate(df, feats, regime_s) for s in all_strategies(self.cfg)]
            # disjoint regimes => at most one active; merge by taking the
            # non-flat target (priority order matches all_strategies()).
            merged = pd.DataFrame(index=df.index)
            merged["target_dir"] = 0.0
            merged["stop_dist"] = np.nan
            merged["note"] = "all strategies flat"
            for st in strat_targets:
                td = pd.to_numeric(st["target_dir"], errors="coerce").fillna(0.0)
                act = (td != 0).values
                merged.loc[act, "target_dir"] = td.values[act].astype(float)
                merged.loc[act, "stop_dist"] = pd.to_numeric(
                    st["stop_dist"], errors="coerce").values[act]
                merged.loc[act, "note"] = st["note"].values[act]
            merged["atr_pct"] = (ind.atr(df, self.cfg["regime"]["atr_window"]) / df["close"])
            merged["close"] = df["close"]
            merged["regime"] = regime_s
            signals[t] = merged
            regimes_out[t] = reg
            ens_store[t] = ens
        return signals, regimes_out, ens_store

    # ------------------------------------------------------------------
    def run(self, data: dict[str, pd.DataFrame]) -> dict:
        signals, regimes, ens_store = self._per_asset_signals(data)
        assets = list(data)
        index = signals[assets[0]].index

        # daily returns matrix (close-to-close)
        ret = pd.DataFrame({a: data[a]["close"].pct_change() for a in assets}).reindex(index)
        dir_df = pd.DataFrame({a: signals[a]["target_dir"] for a in assets}).reindex(index).fillna(0.0)
        stop_df = pd.DataFrame({a: signals[a]["stop_dist"] for a in assets}).reindex(index)
        px_df = pd.DataFrame({a: signals[a]["close"] for a in assets}).reindex(index)
        atrp_df = pd.DataFrame({a: signals[a]["atr_pct"] for a in assets}).reindex(index)

        risk = RiskEngine(self.cfg)
        cap = self.bt["initial_capital"]
        comm = self.bt["commission_bps"] / 1e4

        equity = pd.Series(index=index, dtype=float)
        weights_hist = pd.DataFrame(0.0, index=index, columns=assets)
        w_prev = {a: 0.0 for a in assets}        # weights decided at t-1 (held over t)
        w_prev2 = {a: 0.0 for a in assets}       # decided at t-2 (for turnover)
        trades = []

        for i, dt in enumerate(index):
            # 1) realise PnL for day dt from weights decided yesterday
            if i == 0:
                equity.iloc[i] = cap
            else:
                day_ret = sum(w_prev[a] * (ret[a].iloc[i] if not np.isnan(ret[a].iloc[i]) else 0.0)
                              for a in assets)
                # cost charged on the rebalance executed at start of dt
                cost = 0.0
                for a in assets:
                    dw = abs(w_prev[a] - w_prev2[a])
                    if dw > 0:
                        slip = self._slippage(atrp_df[a].iloc[i - 1])
                        cost += dw * (comm + slip)
                equity.iloc[i] = equity.iloc[i - 1] * (1 + day_ret) - equity.iloc[i - 1] * cost

            # 2) drawdown circuit breaker (uses equity up to dt)
            risk.update_drawdown(equity.iloc[i])

            # 3) decide weights for tomorrow using info up to close(dt)
            ret_window = ret.iloc[max(0, i - self.cfg["risk"]["corr_window"]):i + 1].dropna(how="all")
            dirs = {a: float(dir_df[a].iloc[i]) for a in assets}
            stops = {a: float(stop_df[a].iloc[i]) if not np.isnan(stop_df[a].iloc[i]) else np.nan
                     for a in assets}
            prices = {a: float(px_df[a].iloc[i]) for a in assets}
            w_new = risk.size(dirs, stops, prices, ret_window)

            # no-trade band: ignore tiny weight drift (mostly vol-target wiggle)
            band = self.bt.get("rebalance_band", 0.0)
            if band > 0:
                for a in assets:
                    if abs(w_new[a] - w_prev[a]) < band and np.sign(w_new[a]) == np.sign(w_prev[a]):
                        w_new[a] = w_prev[a]

            # 4) trade log on direction/active changes
            for a in assets:
                if np.sign(w_new[a]) != np.sign(w_prev[a]):
                    trades.append({
                        "date": dt, "asset": a,
                        "action": _action(w_prev[a], w_new[a]),
                        "weight": round(w_new[a], 4),
                        "price": round(prices[a], 4),
                        "regime": signals[a]["regime"].iloc[i],
                        "equity": round(equity.iloc[i], 2),
                        "note": signals[a]["note"].iloc[i],
                    })

            weights_hist.loc[dt] = w_new
            w_prev2 = dict(w_prev)
            w_prev = w_new

        equity.name = "equity"
        dd = equity / equity.cummax() - 1.0
        dd.name = "drawdown"
        regime_panel = pd.DataFrame({a: regimes[a]["regime"] for a in assets})

        return {
            "equity": equity,
            "drawdown": dd,
            "weights": weights_hist,
            "regimes": regime_panel,
            "trades": pd.DataFrame(trades),
            "returns": ret,
            "signals": signals,
            "ensembles": ens_store,
            "halted": risk.halted,
            "n_breaches": risk.n_breaches,
        }

    # ------------------------------------------------------------------
    def _slippage(self, atr_pct: float) -> float:
        if self.bt["slippage_model"] == "vol" and not np.isnan(atr_pct):
            return self.bt["slippage_atr_mult"] * atr_pct
        return self.bt["slippage_bps"] / 1e4


def _action(old: float, new: float) -> str:
    if old == 0 and new > 0:
        return "OPEN_LONG"
    if old == 0 and new < 0:
        return "OPEN_SHORT"
    if new == 0:
        return "CLOSE"
    return "FLIP"
