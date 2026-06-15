"""
Institutional Feature
=====================
Headline: Monte Carlo equity simulation via *block bootstrap* of realised
daily strategy returns (blocks preserve short-term autocorrelation/vol
clustering that an iid bootstrap would destroy). Produces a distribution of
terminal equity, CAGR and max-drawdown plus a fan chart.

Also included: a simple out-of-sample (OOS) split report, so the same run
shows in-sample vs held-out performance.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def monte_carlo(equity: pd.Series, cfg: dict) -> dict:
    runs = cfg["institutional"]["mc_runs"]
    block = cfg["institutional"]["mc_block"]
    ppy = cfg["report"]["periods_per_year"]
    out = Path(cfg["report"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    rets = equity.pct_change().dropna().values
    n = len(rets)
    if n < block * 3:
        return {"error": "not enough data for Monte Carlo"}

    rng = np.random.default_rng(123)
    n_blocks = int(np.ceil(n / block))
    paths = np.empty((runs, n))
    finals, maxdds, cagrs = [], [], []

    for k in range(runs):
        starts = rng.integers(0, n - block, size=n_blocks)
        seq = np.concatenate([rets[s:s + block] for s in starts])[:n]
        eq = np.cumprod(1 + seq)
        paths[k] = eq
        finals.append(eq[-1])
        peak = np.maximum.accumulate(eq)
        maxdds.append((eq / peak - 1).min())
        cagrs.append(eq[-1] ** (ppy / n) - 1)

    pct = lambda arr, p: float(np.percentile(arr, p))
    # fan chart
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    plt.figure(figsize=(11, 5))
    x = np.arange(n)
    plt.fill_between(x, p5, p95, color="steelblue", alpha=0.25, label="5–95%")
    plt.plot(x, p50, color="navy", lw=1.5, label="median")
    plt.plot(x, np.cumprod(1 + rets), color="black", lw=1.2, ls="--", label="realised")
    plt.title(f"Monte Carlo equity ({runs} block-bootstrap paths)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "monte_carlo.png", dpi=120); plt.close()

    return {
        "runs": runs, "block": block,
        "final_multiple": {"p5": round(pct(finals, 5), 3),
                           "p50": round(pct(finals, 50), 3),
                           "p95": round(pct(finals, 95), 3)},
        "cagr": {"p5": round(pct(cagrs, 5), 4),
                 "p50": round(pct(cagrs, 50), 4),
                 "p95": round(pct(cagrs, 95), 4)},
        "max_drawdown": {"p5": round(pct(maxdds, 5), 4),
                         "p50": round(pct(maxdds, 50), 4),
                         "p95": round(pct(maxdds, 95), 4)},
        "prob_loss": round(float(np.mean(np.array(finals) < 1.0)), 3),
    }


def oos_report(results: dict, cfg: dict) -> dict:
    """In-sample vs out-of-sample metrics on the realised equity curve."""
    from ..report.performance import equity_metrics
    eq = results["equity"]
    frac = cfg["institutional"]["oos_split"]
    cut = int(len(eq) * frac)
    is_eq = eq.iloc[:cut]
    oos_eq = eq.iloc[cut:] / eq.iloc[cut] * cfg["backtest"]["initial_capital"]
    return {
        "split_date": str(eq.index[cut].date()),
        "in_sample": equity_metrics(is_eq, cfg),
        "out_of_sample": equity_metrics(oos_eq, cfg),
    }
