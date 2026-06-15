"""
Research Dashboard  (Priority #4)
================================
Streamlit + Plotly dashboard over the artifacts written to outputs/report/.

Run:
    pip install streamlit plotly
    streamlit run src/dashboard/streamlit_app.py

Panels: regime timeline, equity curve (vs SPY), drawdown, rolling Sharpe,
portfolio weights over time, and strategy/regime allocation changes.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except Exception as e:                       # allow import without the deps
    st = None

OUT = Path("outputs/report")
REGIME_COLORS = {"TREND_UP": "#2ca02c", "TREND_DOWN": "#d62728",
                 "RANGE": "#7f7f7f", "HIGH_VOL_CRASH": "#9467bd"}


def _load():
    eq = pd.read_csv(OUT / "equity.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv(OUT / "regimes.csv", index_col=0, parse_dates=True)
    summary = json.loads((OUT / "summary.json").read_text())
    trades = pd.read_csv(OUT / "trades.csv", parse_dates=["date"]) \
        if (OUT / "trades.csv").exists() else pd.DataFrame()
    return eq, regimes, summary, trades


def main():
    if st is None:
        print("streamlit/plotly not installed. Run: pip install streamlit plotly")
        return
    st.set_page_config(page_title="Market Regime Engine", layout="wide")
    st.title("Market Regime + Strategy Engine — Research Dashboard")

    eq, regimes, summary, trades = _load()

    # headline metrics
    s = summary["strategy"]
    c = st.columns(6)
    c[0].metric("CAGR", f"{s['cagr']:.1%}")
    c[1].metric("Sharpe", s["sharpe"])
    c[2].metric("Sortino", s["sortino"])
    c[3].metric("Calmar", s["calmar"])
    c[4].metric("Max DD", f"{s['max_drawdown']:.1%}")
    c[5].metric("Profit factor", s["profit_factor"])

    # equity curve
    st.subheader("Equity curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq["equity"], name="Regime Engine"))
    st.plotly_chart(fig, use_container_width=True)

    # drawdown
    st.subheader("Drawdown")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq["drawdown"], fill="tozeroy",
                             line_color="crimson", name="Drawdown"))
    st.plotly_chart(fig, use_container_width=True)

    # rolling sharpe
    st.subheader("Rolling Sharpe (126d)")
    r = eq["equity"].pct_change()
    roll = (r.rolling(126).mean() / r.rolling(126).std()) * np.sqrt(252)
    fig = px.line(x=roll.index, y=roll.values, labels={"y": "Sharpe", "x": ""})
    st.plotly_chart(fig, use_container_width=True)

    # regime timeline (benchmark)
    st.subheader("Regime timeline")
    asset = st.selectbox("Asset", list(regimes.columns))
    rr = regimes[asset]
    fig = go.Figure()
    for reg, col in REGIME_COLORS.items():
        mask = rr == reg
        fig.add_trace(go.Scatter(x=rr.index[mask], y=[reg] * mask.sum(),
                                 mode="markers", marker=dict(color=col, size=4),
                                 name=reg))
    st.plotly_chart(fig, use_container_width=True)

    # trades
    if not trades.empty:
        st.subheader("Trade log")
        st.dataframe(trades.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()
