"""
Walk-Forward Optimization  (institutional feature)
=================================================
Rolling-window, anchored-free walk-forward:

  [---- train ----][- test -]
          [---- train ----][- test -]
                  [---- train ----][- test -]

On each TRAIN window we grid-search a small parameter set and keep the combo
with the best in-sample Sharpe; we then trade that combo UNSEEN on the
following TEST window. Concatenating the test windows yields a fully
out-of-sample equity curve — the honest estimate of live performance.

HMM is disabled inside the search for speed (the causal rule ensemble still
drives regimes); flip cfg['regime']['hmm_enabled'] back on for a final pass.
"""
from __future__ import annotations
import copy
import itertools
import numpy as np
import pandas as pd

from ..backtest.engine import Backtester
from ..report.performance import equity_metrics


PARAM_GRID = {
    "mean_reversion.z_entry": [1.5, 2.0, 2.5],
    "trend.stop_atr": [2.0, 3.0],
}


def _apply_params(cfg: dict, combo: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["regime"]["hmm_enabled"] = False          # speed
    for key, val in combo.items():
        grp, name = key.split(".")
        c["strategy"][grp][name] = val
    return c


def _slice(data: dict, start, end) -> dict:
    return {a: df.loc[start:end] for a, df in data.items()}


def _sharpe(equity: pd.Series, cfg: dict) -> float:
    m = equity_metrics(equity, cfg)
    return m.get("sharpe", -np.inf)


def walk_forward(data: dict, cfg: dict, train_years: int = 3,
                 test_years: int = 1) -> dict:
    idx = next(iter(data.values())).index
    ppy = cfg["report"]["periods_per_year"]
    train_n, test_n = train_years * ppy, test_years * ppy

    combos = [dict(zip(PARAM_GRID, v)) for v in itertools.product(*PARAM_GRID.values())]
    oos_curves, fold_log = [], []

    start = 0
    fold = 0
    while start + train_n + test_n <= len(idx):
        tr0, tr1 = idx[start], idx[start + train_n - 1]
        te0, te1 = idx[start + train_n], idx[min(start + train_n + test_n - 1, len(idx) - 1)]
        train_data = _slice(data, tr0, tr1)
        test_data = _slice(data, te0, te1)

        # ---- grid search on train ----
        best, best_sharpe = combos[0], -np.inf
        for combo in combos:
            ccfg = _apply_params(cfg, combo)
            res = Backtester(ccfg).run(train_data)
            s = _sharpe(res["equity"], ccfg)
            if s > best_sharpe:
                best_sharpe, best = s, combo

        # ---- apply best combo OOS on test ----
        tcfg = _apply_params(cfg, best)
        test_res = Backtester(tcfg).run(test_data)
        oos_curves.append(test_res["equity"])
        fold_log.append({
            "fold": fold,
            "train": f"{tr0.date()}..{tr1.date()}",
            "test": f"{te0.date()}..{te1.date()}",
            "best_params": best,
            "train_sharpe": round(float(best_sharpe), 3),
            "test_sharpe": round(float(_sharpe(test_res["equity"], tcfg)), 3),
        })
        start += test_n
        fold += 1

    # stitch OOS equity (chain returns across folds)
    stitched = []
    base = cfg["backtest"]["initial_capital"]
    for eq in oos_curves:
        r = eq.pct_change().fillna(0.0)
        seg = base * (1 + r).cumprod()
        base = seg.iloc[-1]
        stitched.append(seg)
    oos_equity = pd.concat(stitched) if stitched else pd.Series(dtype=float)

    summary = {
        "n_folds": len(fold_log),
        "folds": fold_log,
        "oos_metrics": equity_metrics(oos_equity, cfg) if len(oos_equity) else {},
        "avg_test_sharpe": round(float(np.mean([f["test_sharpe"] for f in fold_log])), 3)
        if fold_log else 0.0,
    }
    return {"summary": summary, "oos_equity": oos_equity}
