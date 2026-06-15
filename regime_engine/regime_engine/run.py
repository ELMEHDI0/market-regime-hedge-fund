#!/usr/bin/env python3
"""
Market Regime + Strategy Engine — full pipeline.

    python run.py                 # full pipeline (config.yaml)
    python run.py --config x.yaml
    python run.py --explain QQQ    # regime explanations for one asset
    python run.py --walkforward    # also run walk-forward optimization (slower)

Pipeline: data -> regime -> strategy -> risk -> portfolio -> backtest
          -> report -> factor analysis -> stress test -> (walk-forward)
          -> experiment tracking.  Dashboard: streamlit run src/dashboard/streamlit_app.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.loader import load_market_data
from src.backtest.engine import Backtester
from src.report.performance import build_report
from src.institutional.monte_carlo import monte_carlo, oos_report
from src.portfolio.portfolio import all_static_allocations, regime_portfolio_backtest
from src.analytics.factor_analysis import run_factor_analysis
from src.analytics.stress_testing import stress_test
from src.analytics.experiment_tracking import ExperimentTracker
from src.regime.detectors import REGIMES


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def explain_regimes(results: dict, asset: str) -> str:
    ens = results["ensembles"][asset]
    reg_frame = results["regimes_full"][asset]
    reg = reg_frame["regime"]
    lines = [f"\n################ REGIME EXPLANATIONS — {asset} ################"]
    for label in REGIMES:
        days = reg.index[reg == label]
        if len(days) == 0:
            lines.append(f"\n[{label}] never occurred for {asset}.")
            continue
        lines.append("\n" + ens.explain(reg_frame, days[-1]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--explain", default=None)
    ap.add_argument("--walkforward", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = Path(cfg["report"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    print("=" * 70 + "\nMARKET REGIME + STRATEGY ENGINE\n" + "=" * 70)

    with ExperimentTracker(cfg) as tracker:
        tracker.log_params({"data": {"tickers": ",".join(cfg["data"]["tickers"])},
                            "risk": cfg["risk"], "regime": cfg["regime"]})

        # 1) data
        data = load_market_data(cfg)

        # 2-5) backtest (regime + strategy + risk + costs)
        results = Backtester(cfg).run(data)
        results["regimes_full"] = {a: results["ensembles"][a].run(data[a]) for a in data}

        # 6) performance report
        summary = build_report(results, cfg)
        tracker.log_metrics(summary["strategy"])
        print("\n----- PERFORMANCE (strategy vs SPY buy & hold) -----")
        print(json.dumps(summary, indent=2))

        # Priority #1: portfolio construction
        rets = pd.DataFrame({a: data[a]["close"].pct_change() for a in data}).dropna()
        cov = rets.cov() * cfg["report"]["periods_per_year"]
        static_alloc = all_static_allocations(cov, cfg)
        market_regime = results["regimes"][cfg["data"]["benchmark"]]
        reg_port = regime_portfolio_backtest(data, market_regime, cfg)
        from src.report.performance import equity_metrics
        portfolio_out = {
            "static_allocations": {k: {a: round(w, 3) for a, w in v.items()}
                                   for k, v in static_alloc.items()},
            "regime_portfolio_metrics": equity_metrics(reg_port["equity"], cfg),
        }
        (out / "portfolio.json").write_text(json.dumps(portfolio_out, indent=2))
        reg_port["weights"].to_csv(out / "portfolio_weights.csv")
        print("\n----- PORTFOLIO CONSTRUCTION -----")
        print(json.dumps(portfolio_out, indent=2))

        # Priority #2: factor analysis
        factors = run_factor_analysis(results, data, cfg)
        (out / "factor_analysis.json").write_text(json.dumps(factors, indent=2))
        tracker.log_metrics({k: v for k, v in factors.items() if isinstance(v, (int, float))})
        print("\n----- FACTOR EXPOSURE ANALYSIS -----")
        print(json.dumps(factors, indent=2))

        # Priority #3: stress testing
        stress = stress_test(results["equity"], cfg)
        (out / "stress_test.json").write_text(json.dumps(stress, indent=2))
        print("\n----- MONTE CARLO STRESS TEST -----")
        print(json.dumps(stress, indent=2))

        # institutional: monte carlo + OOS split
        mc = monte_carlo(results["equity"], cfg)
        oos = oos_report(results, cfg)
        (out / "monte_carlo.json").write_text(json.dumps(mc, indent=2))
        (out / "oos.json").write_text(json.dumps(oos, indent=2))

        # walk-forward (optional)
        if args.walkforward:
            from src.optimization.walk_forward import walk_forward
            print("\n----- WALK-FORWARD OPTIMIZATION (this takes a while) -----")
            wf = walk_forward(data, cfg)
            (out / "walk_forward.json").write_text(json.dumps(wf["summary"], indent=2))
            wf["oos_equity"].to_frame("oos_equity").to_csv(out / "walk_forward_oos.csv")
            tracker.log_metrics({"wf_avg_test_sharpe": wf["summary"]["avg_test_sharpe"]})
            print(json.dumps(wf["summary"], indent=2))

        # explainability
        target = args.explain or cfg["data"]["benchmark"]
        expl = explain_regimes(results, target)
        (out / f"regime_explanations_{target}.txt").write_text(expl)
        print(expl)

        for f in ["equity_curve.png", "drawdown_curve.png", "summary.json"]:
            tracker.log_artifact(str(out / f))

    print(f"\nDONE. Artifacts in: {out}/")
    print("Dashboard: streamlit run src/dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
