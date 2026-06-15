"""
Performance Report
==================
Metrics: Sharpe, Sortino, Calmar, win rate, profit factor, max drawdown,
exposure time, CAGR — for the strategy AND for buy & hold SPY.

Also writes: equity_curve.png, drawdown_curve.png, trades.csv, regimes.csv,
summary.json and summary.md into the report directory.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ann(cfg):  # periods per year
    return cfg["report"]["periods_per_year"]


def equity_metrics(equity: pd.Series, cfg: dict) -> dict:
    ppy = _ann(cfg)
    rf = cfg["report"]["rf_annual"]
    ret = equity.pct_change().dropna()
    if ret.std() == 0 or len(ret) < 2:
        return {k: 0.0 for k in ["cagr", "sharpe", "sortino", "calmar",
                                 "max_drawdown", "total_return", "vol_annual"]}
    years = len(ret) / ppy
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    excess = ret - rf / ppy
    sharpe = excess.mean() / ret.std() * np.sqrt(ppy)
    downside = ret[ret < 0].std()
    sortino = excess.mean() / downside * np.sqrt(ppy) if downside and downside > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else 0.0
    return {
        "cagr": round(float(cagr), 4),
        "total_return": round(float(total), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "calmar": round(float(calmar), 3),
        "max_drawdown": round(float(dd), 4),
        "vol_annual": round(float(ret.std() * np.sqrt(ppy)), 4),
    }


def trade_metrics(weights: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Win rate, profit factor, #trades, exposure — from per-asset round trips."""
    contrib = weights.shift(1) * returns.reindex(weights.index)
    trade_pnls = []
    for a in weights.columns:
        w = weights[a].fillna(0.0)
        sign = np.sign(w)
        c = contrib[a].fillna(0.0)
        cur, start = 0, None
        for i in range(len(w)):
            s = sign.iloc[i]
            if s != cur:
                if cur != 0 and start is not None:
                    trade_pnls.append(c.iloc[start:i].sum())
                cur, start = s, i
        if cur != 0 and start is not None:
            trade_pnls.append(c.iloc[start:].sum())

    trade_pnls = [p for p in trade_pnls if p != 0]
    n = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / n if n else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else (np.inf if wins else 0.0)
    exposure = float((weights.abs().sum(axis=1) > 1e-9).mean())
    return {
        "n_trades": n,
        "win_rate": round(win_rate, 3),
        "profit_factor": round(float(pf), 3) if np.isfinite(pf) else "inf",
        "exposure_time": round(exposure, 3),
    }


def buy_and_hold(returns: pd.DataFrame, benchmark: str, equity0: float) -> pd.Series:
    r = returns[benchmark].fillna(0.0)
    return (equity0 * (1 + r).cumprod()).rename("spy_buyhold")


# --------------------------------------------------------------------------
def build_report(results: dict, cfg: dict) -> dict:
    out = Path(cfg["report"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    eq = results["equity"]
    bh = buy_and_hold(results["returns"], cfg["data"]["benchmark"],
                      cfg["backtest"]["initial_capital"])

    strat = {**equity_metrics(eq, cfg),
             **trade_metrics(results["weights"], results["returns"])}
    bench = equity_metrics(bh, cfg)

    # ---- plots ----
    plt.figure(figsize=(11, 5))
    plt.plot(eq.index, eq.values, label="Regime Engine", lw=1.6)
    plt.plot(bh.index, bh.values, label="Buy & Hold SPY", lw=1.2, alpha=0.8)
    plt.title("Equity Curve"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "equity_curve.png", dpi=120); plt.close()

    plt.figure(figsize=(11, 3.2))
    plt.fill_between(results["drawdown"].index, results["drawdown"].values, 0,
                     color="crimson", alpha=0.4)
    plt.title("Drawdown"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "drawdown_curve.png", dpi=120); plt.close()

    # ---- tables ----
    results["trades"].to_csv(out / "trades.csv", index=False)
    results["regimes"].to_csv(out / "regimes.csv")
    eq.to_frame().join(results["drawdown"]).to_csv(out / "equity.csv")

    summary = {"strategy": strat, "spy_buy_hold": bench,
               "drawdown_breaches": int(results.get("n_breaches", 0)),
               "halted_at_end": bool(results["halted"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(_markdown(summary, cfg))
    print("[report] written to", out)
    return summary


def _markdown(s: dict, cfg: dict) -> str:
    st, bh = s["strategy"], s["spy_buy_hold"]
    rows = [
        ("CAGR", st["cagr"], bh["cagr"]),
        ("Total Return", st["total_return"], bh["total_return"]),
        ("Sharpe", st["sharpe"], bh["sharpe"]),
        ("Sortino", st["sortino"], bh["sortino"]),
        ("Calmar", st["calmar"], bh["calmar"]),
        ("Max Drawdown", st["max_drawdown"], bh["max_drawdown"]),
        ("Annual Vol", st["vol_annual"], bh["vol_annual"]),
    ]
    lines = ["# Performance Report", "",
             "| Metric | Regime Engine | Buy & Hold SPY |",
             "|---|---|---|"]
    for n, a, b in rows:
        lines.append(f"| {n} | {a} | {b} |")
    lines += ["",
              f"**Trades:** {st['n_trades']}  |  **Win rate:** {st['win_rate']}  |  "
              f"**Profit factor:** {st['profit_factor']}  |  "
              f"**Exposure:** {st['exposure_time']}",
              f"**Drawdown circuit-breaker hits:** {s['drawdown_breaches']}  "
              f"(halted at end: {s['halted_at_end']})"]
    return "\n".join(lines)
