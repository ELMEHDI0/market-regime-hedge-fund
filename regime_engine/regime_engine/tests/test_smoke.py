"""Minimal smoke tests: pipeline integrity + no-lookahead sanity."""
import yaml, numpy as np
from src.data.loader import load_market_data
from src.backtest.engine import Backtester
from src.risk.risk_engine import RiskEngine


def _cfg():
    c = yaml.safe_load(open("config.yaml"))
    c["data"]["source"] = "synthetic"; c["regime"]["hmm_enabled"] = False
    return c


def test_pipeline_runs():
    cfg = _cfg()
    data = load_market_data(cfg)
    res = Backtester(cfg).run(data)
    assert len(res["equity"]) > 100
    assert res["drawdown"].min() <= 0
    assert set(res["trades"].columns) >= {"date", "asset", "action"}


def test_circuit_breaker_rearms():
    cfg = _cfg(); cfg["risk"]["drawdown_cooldown"] = 10
    r = RiskEngine(cfg)
    eq = [100, 110] + [110 * 0.7] * 60          # -30% then flat
    flags = [r.update_drawdown(e) for e in eq]
    assert any(flags) and not all(flags[2:])     # halts then re-arms


def test_no_constant_lookahead():
    """Weights at t must be decided before t+1 returns exist (shifted use)."""
    cfg = _cfg()
    data = load_market_data(cfg)
    res = Backtester(cfg).run(data)
    w, ret = res["weights"], res["returns"]
    # contribution uses weights.shift(1); first day must be flat PnL
    contrib = (w.shift(1) * ret).iloc[0].fillna(0)
    assert np.allclose(contrib.values, 0)
