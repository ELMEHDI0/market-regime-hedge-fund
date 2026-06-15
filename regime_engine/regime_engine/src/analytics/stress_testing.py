"""
Monte Carlo Stress Testing  (Priority #3)
========================================
Block-bootstrap the strategy's realised daily returns into many alternate
histories and measure tail risk:

  * Probability of Ruin     (terminal equity below a ruin threshold)
  * Expected Max Drawdown    (mean of per-path worst drawdowns)
  * Worst-Case Return        (min terminal across paths)
  * VaR 95 / 99              (daily, historical)
  * CVaR 95 / 99             (expected shortfall beyond VaR)

Runs at several simulation counts (e.g. 1,000 and 5,000) to show convergence.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _var_cvar(daily: np.ndarray, level: float) -> tuple[float, float]:
    q = np.percentile(daily, (1 - level) * 100)
    cvar = daily[daily <= q].mean() if (daily <= q).any() else q
    return float(q), float(cvar)


def _simulate(daily: np.ndarray, runs: int, block: int, rng) -> dict:
    n = len(daily)
    n_blocks = int(np.ceil(n / block))
    finals, maxdds = [], []
    for _ in range(runs):
        starts = rng.integers(0, n - block, size=n_blocks)
        seq = np.concatenate([daily[s:s + block] for s in starts])[:n]
        eq = np.cumprod(1 + seq)
        finals.append(eq[-1])
        peak = np.maximum.accumulate(eq)
        maxdds.append(float((eq / peak - 1).min()))
    return {"finals": np.array(finals), "maxdds": np.array(maxdds)}


def stress_test(equity: pd.Series, cfg: dict, ruin_threshold: float = 0.5) -> dict:
    out = Path(cfg["report"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    daily = equity.pct_change().dropna().values
    if len(daily) < 50:
        return {"error": "not enough data"}
    block = cfg["institutional"]["mc_block"]
    rng = np.random.default_rng(2024)

    report = {}
    sim_counts = sorted({1000, 5000, int(cfg["institutional"]["mc_runs"])})
    last = None
    for runs in sim_counts:
        sim = _simulate(daily, runs, block, rng)
        finals, maxdds = sim["finals"], sim["maxdds"]
        v95, c95 = _var_cvar(daily, 0.95)
        v99, c99 = _var_cvar(daily, 0.99)
        report[f"runs_{runs}"] = {
            "prob_of_ruin": round(float((finals < ruin_threshold).mean()), 4),
            "expected_max_drawdown": round(float(maxdds.mean()), 4),
            "worst_max_drawdown": round(float(maxdds.min()), 4),
            "worst_case_terminal_multiple": round(float(finals.min()), 3),
            "median_terminal_multiple": round(float(np.median(finals)), 3),
            "VaR_95_daily": round(v95, 4),
            "CVaR_95_daily": round(c95, 4),
            "VaR_99_daily": round(v99, 4),
            "CVaR_99_daily": round(c99, 4),
        }
        last = sim

    # distribution plot from the largest run
    plt.figure(figsize=(11, 4))
    plt.hist(last["finals"], bins=60, color="slateblue", alpha=0.8)
    plt.axvline(ruin_threshold, color="crimson", ls="--",
                label=f"ruin < {ruin_threshold:g}×")
    plt.axvline(1.0, color="black", ls=":", label="break-even")
    plt.title("Stress test — terminal equity multiple distribution")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "stress_test.png", dpi=120); plt.close()

    report["ruin_threshold"] = ruin_threshold
    return report
