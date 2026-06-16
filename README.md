Market Regime + Strategy Engine — Mini Quant Research System

A modular quantitative research framework that classifies market regimes, allocates capital across regime-dependent strategies, and evaluates performance using institutional-grade backtesting, risk controls, and stress testing.

The system is designed for research reproducibility, explainability, and walk-forward validation, not curve-fitted performance claims.

Performance Snapshot
Metric	Value
CAGR	X%
Sharpe Ratio	X.XX
Max Drawdown	-X%
Sortino Ratio	X.XX
Win Rate	X%
Benchmark (SPY)	+X%

Equity and drawdown charts are generated in /outputs/report/.

Core Idea

Markets behave differently depending on regime conditions. This system:

Detects market regimes (trend, range, high volatility, crash)
Activates only strategies suited to current regime
Sizes risk dynamically
Evaluates results with no-lookahead backtesting

If the system cannot explain a decision, it is not allowed to trade it.

System Pipeline
Data → Regime Detection → Strategy Selection → Risk Engine → Portfolio Construction → Backtest → Reporting
Architecture
regime_engine/
├── run.py                  # main orchestrator
├── config.yaml             # all parameters centralized
├── src/
│   ├── data/loader.py      # ingestion + cleaning + alignment
│   ├── regime/             # regime detection logic
│   ├── strategies/         # trading strategies
│   ├── risk/               # position sizing + risk controls
│   ├── portfolio/          # allocation logic
│   ├── backtest/           # no-lookahead execution engine
│   ├── report/             # performance metrics
│   ├── analytics/          # factor + stress + experiments
│   ├── optimization/       # walk-forward optimization
│   └── dashboard/          # Streamlit UI
└── outputs/
    ├── report/
    ├── trades.csv
    ├── equity_curve.png
    └── stress tests
Data Layer

Assets:

SPY, QQQ, GLD, BTC-USD, EURUSD

Processing:

Clean OHLCV data
Remove invalid bars
Align all assets to a unified trading calendar
Store in Parquet (or SQLite fallback)

Fallback:

Synthetic regime generator ensures full pipeline execution when live data is unavailable
(used for validation only, not performance evaluation)
Regime Detection

Market states:

TREND_UP
TREND_DOWN
RANGE
HIGH_VOL_CRASH
Detectors
Detector	Method	Causal
Trend Regime	MA + ATR + ADX + drawdown	Yes
Volatility Regime	EWMA volatility model	Yes
HMM Regime	Gaussian HMM on returns	No (research only)
Ensemble Logic
Crash regime overrides all signals
Otherwise majority vote
Fully explainable decision output per timestamp
Strategies
Strategy	Active Regime	Logic
Trend Following	TREND_UP / DOWN	MA momentum + ATR trailing stop
Mean Reversion	RANGE	Bollinger z-score reversion
Vol Breakout	HIGH_VOL_CRASH	Donchian breakout system

Each trade includes:

Entry reason
Exit condition
Stop distance (ATR-based)
Risk Engine

Hard constraints:

Position risk: 0.5–1% per trade
Volatility targeting
Correlation reduction (exposure haircut on correlated bets)
Gross/net exposure caps
Drawdown circuit breaker (auto halt + cooldown)

No emotional discretion. Only rules.

Backtesting Engine

Guarantees:

No lookahead bias
No repainting indicators
Realistic slippage + commission modeling
Execution at t+1 based on t signals

Outputs:

Equity curve
Drawdown curve
Trade-level dataset
Portfolio Construction

Supported methods:

Equal weight
Inverse volatility
Risk parity (ERC)
Maximum diversification (optimization-based)
Regime-weighted allocation
Analytics & Stress Testing

Includes:

Monte Carlo equity simulation (bootstrapped paths)
VaR / CVaR (95% / 99%)
Probability of ruin
Worst-case drawdown distribution
Factor exposure (beta, alpha, IR, tracking error vs SPY)
Walk-Forward Optimization
Rolling train/test windows
Parameter grid search on training set
Evaluation on unseen test set
Final stitched out-of-sample equity curve

This is the closest approximation to real-world forward performance.

Outputs

Generated artifacts:

equity_curve.png
drawdown_curve.png
monte_carlo.png
stress_test.json
trades.csv
regimes.csv
summary.json
walk_forward.json
Limitations
Synthetic mode is not investable signal
HMM is non-causal unless run in walk-forward mode
Slippage/costs are modeled, not exchange exact
Research system, not live trading infrastructure
Summary

A modular, explainable quantitative trading system built around:

regime detection
risk-first execution
walk-forward validation
institutional reporting standards

Designed for research, not storytelling.
