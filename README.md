Market Regime + Strategy Engine — Mini Hedge Fund Core
A research-grade, end-to-end systematic trading stack: it pulls multi-asset data, classifies the market into regimes, runs regime-gated strategies, sizes them through a hard-constraint risk engine, backtests with no lookahead and realistic costs, and produces an institutional performance, factor and stress report — all from one command.

Design philosophy — explainability is mandatory. Every regime label and every strategy on/off decision carries a machine-readable reason. If the system can't tell you why a regime exists or why a strategy is active, it is not allowed to trade it.

One command
pip install -r requirements.txt
python run.py                 # full pipeline -> outputs/report/
python run.py --walkforward   # also run walk-forward optimization
python run.py --explain QQQ    # print regime rationale for one asset
streamlit run src/dashboard/streamlit_app.py   # research dashboard
Live data is pulled with yfinance. If the network is unavailable (e.g. CI / sandbox), the loader transparently falls back to a regime-switching synthetic generator so the whole pipeline still runs and is validated end-to-end. The source actually used is recorded in outputs/market_data/_SOURCE.txt.

⚠️ On synthetic data the P&L is illustrative only — it exists to prove the machinery (regime logic, risk controls, costs, reports) executes correctly. Point the loader at real data (data.source: yfinance) for meaningful results.

Architecture
regime_engine/
├── run.py                      # one-command orchestrator
├── config.yaml                 # every parameter lives here
├── src/
│   ├── data/loader.py          # fetch + clean + align + store (Parquet/SQLite)
│   ├── regime/
│   │   ├── indicators.py       # causal ATR / ADX / vol / percentile / drawdown
│   │   └── detectors.py        # 3 detectors + explainable ENSEMBLE  (core edge)
│   ├── strategies/strategies.py# trend / mean-reversion / vol-breakout (regime-gated)
│   ├── risk/risk_engine.py     # vol sizing, risk budget, corr haircut, DD breaker
│   ├── portfolio/portfolio.py  # equal-wt / risk-parity / max-div / regime allocations
│   ├── backtest/engine.py      # no-lookahead portfolio loop + costs + slippage
│   ├── report/performance.py   # Sharpe/Sortino/Calmar/PF/... vs SPY buy & hold
│   ├── analytics/
│   │   ├── factor_analysis.py  # alpha, beta, IR, tracking error, factor betas
│   │   ├── stress_testing.py   # prob of ruin, VaR/CVaR, expected max DD
│   │   └── experiment_tracking.py # MLflow logging (graceful fallback)
│   ├── institutional/monte_carlo.py # MC equity sim + out-of-sample split
│   ├── optimization/walk_forward.py # rolling train/test grid search (OOS)
│   └── dashboard/streamlit_app.py   # Streamlit + Plotly research UI
└── outputs/                    # data + report artifacts
1. Data layer
Daily OHLCV for SPY, QQQ, GLD, BTC-USD, EURUSD. Cleaning strips timezones, removes corrupt/non-positive bars, forward-fills only short gaps (≤3 days so crypto-vs-equity calendars aren't fabricated) and aligns every asset onto one trading-day calendar anchored to SPY. Stored as Parquet (default) or SQLite.

2. Regime detection — the core edge module
Four regimes: TREND_UP, TREND_DOWN, RANGE, HIGH_VOL_CRASH. Three independent detectors:

Detector	Method	Causal?
RuleRegime	MA stack + ATR + ADX + drawdown	✅ drives live trading
VolRegime	EWMA conditional vol (RiskMetrics λ=0.94, a GARCH-like proxy) clustering	✅
HMMRegime	4-state Gaussian HMM on returns/vol features	⚠️ analytical (see lookahead)
They are combined by an explainable ensemble: a CRASH flag from the causal rule engine is a veto (capital preservation can't be out-voted); otherwise a majority vote, with the causal rule engine breaking ties. ensemble.explain(date) prints the full rationale (feature values + every detector's vote).

3. Strategy layer (regime-gated)
Strategy	Active regime	Logic
Trend following	TREND_UP / TREND_DOWN	ride fast-MA with ATR trailing stop
Mean reversion	RANGE	fade Bollinger z-score (±2σ), exit at mean
Volatility breakout	HIGH_VOL_CRASH	Donchian channel break (shifted, no repaint)
Each emits an entry, an exit and an ATR-based stop distance, plus a reason string saying why it is active or inactive for that bar.

4. Risk engine
Risk per trade 0.5–1% — positions sized so a stop-out costs ≈ the budget.
Volatility targeting toward a constant portfolio vol.
Correlation reduction — haircut weights of strongly co-moving, same-sign positions (don't take one bet five times).
Exposure caps — per-asset and total gross.
Drawdown circuit breaker (hard stop) — flatten + halt on a max-drawdown breach measured from the all-time high-water mark; re-arms after a cooldown (set drawdown_cooldown: 0 for a permanent stop).
5. Backtesting engine
No lookahead: signals on bar t use only data up to t; weights decided at close(t) are applied to the t→t+1 return. No repainting: Donchian / percentile / walk-forward-HMM channels are shifted. Costs: commission (bps per side on turnover) + fixed or volatility-based slippage (k·ATR%), with a no-trade band so vol-target wiggle doesn't churn. Outputs the equity curve, drawdown curve and full trade list.

6. Performance report
Sharpe, Sortino, Calmar, win rate, profit factor, max drawdown, exposure time, CAGR — for the strategy and benchmarked against buy & hold SPY — plus PNG equity/drawdown charts and CSV tables.

7. Institutional features
Monte Carlo block-bootstrap equity simulation (fan chart + percentiles).
Out-of-sample split report (in-sample vs held-out).
Walk-forward optimization (--walkforward): rolling train/test, grid search on each train window, traded unseen on the next test window, stitched into a fully out-of-sample curve — the honest live estimate.
Added layers (institutional polish)
Portfolio construction — equal weight, inverse-vol, risk parity (equal risk contribution), maximum diversification (SLSQP), vol targeting, and regime-dependent strategic allocations (e.g. bull 60/20/10/10, crisis GLD/cash/bonds).
Factor exposure analysis — alpha, market beta, factor betas (MKT / MOM / LOW-VOL / CARRY), R², Information Ratio and Tracking Error vs SPY. (SIZE/VALUE pluggable via external Fama-French/AQR columns.)
Monte Carlo stress testing — 1k/5k sims: probability of ruin, expected & worst max drawdown, worst-case return, VaR/CVaR 95 & 99.
Research dashboard — Streamlit + Plotly: regime timeline, equity, drawdown, rolling Sharpe, portfolio weights, allocation changes.
Experiment tracking — MLflow logging of params, metrics and artifacts (degrades to a local JSON log if MLflow isn't present).
A note on the HMM and lookahead
Fitting an HMM on the full sample uses future data, so in analytical mode the HMM is an overlay for research/labelling and does not drive fills. The causal rule + vol ensemble drives trading. Set regime.hmm_mode: walkforward for an expanding-window, causal HMM you can safely trade.

Outputs (outputs/report/)
equity_curve.png, drawdown_curve.png, monte_carlo.png, stress_test.png, summary.{json,md}, trades.csv, regimes.csv, equity.csv, portfolio.json, factor_analysis.json, stress_test.json, monte_carlo.json, oos.json, walk_forward.json, experiment_log.json, regime_explanations_<ASSET>.txt.

Limitations
Synthetic-data P&L is not investment signal. FX/crypto calendar alignment is simplified. The HMM analytical mode is non-causal by construction. Costs/slippage are modelled, not exchange-exact. This is research/educational software, not investment advice.
