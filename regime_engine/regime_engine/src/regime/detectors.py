"""
Regime Detection — the core edge module.
=========================================
Four regimes:  TREND_UP, TREND_DOWN, RANGE, HIGH_VOL_CRASH

Three independent detectors, then an explainable ensemble:

  1. RuleRegime  (MA + ATR + ADX)   -- causal, drives live trading
  2. VolRegime   (vol clustering)   -- EWMA / GARCH-like vol state
  3. HMMRegime   (Gaussian HMM)      -- analytical overlay (see lookahead note)

Every detector returns, for each bar, a label AND a human-readable reason.
The brutal requirement — "explain why each regime exists" — is met by the
`.explain(date)` method on the ensemble.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import indicators as ind

TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"
CRASH = "HIGH_VOL_CRASH"
REGIMES = [TREND_UP, TREND_DOWN, RANGE, CRASH]


# =====================================================================
# Shared feature frame
# =====================================================================
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    r = cfg["regime"]
    f = pd.DataFrame(index=df.index)
    f["close"] = df["close"]
    f["ma_fast"] = df["close"].rolling(r["ma_fast"]).mean()
    f["ma_slow"] = df["close"].rolling(r["ma_slow"]).mean()
    f["atr"] = ind.atr(df, r["atr_window"])
    f["atr_pct"] = f["atr"] / df["close"]
    f["adx"] = ind.adx(df, r["adx_window"])
    f["rvol"] = ind.realised_vol(df["close"], r["vol_window"], cfg["report"]["periods_per_year"])
    f["vol_pctile"] = ind.rolling_percentile(f["rvol"], r["vol_lookback_pctile"])
    f["dd_1m"] = ind.rolling_drawdown(df["close"], 21)
    f["ret"] = np.log(df["close"]).diff()
    return f


# =====================================================================
# 1. Rule-based regime (MA + ATR + ADX). CAUSAL — used by the backtest.
# =====================================================================
class RuleRegime:
    name = "rule_ma_atr"

    def __init__(self, cfg: dict):
        self.r = cfg["regime"]

    def classify(self, f: pd.DataFrame) -> pd.DataFrame:
        r = self.r
        label = pd.Series(index=f.index, dtype="object")
        reason = pd.Series(index=f.index, dtype="object")

        crash = (f["vol_pctile"] >= r["crash_vol_pctile"]) | (f["dd_1m"] <= r["crash_drawdown"])
        up = (f["ma_fast"] > f["ma_slow"]) & (f["adx"] >= r["adx_trend_min"])
        down = (f["ma_fast"] < f["ma_slow"]) & (f["adx"] >= r["adx_trend_min"])
        rng = f["adx"] <= r["adx_range_max"]

        for dt in f.index:
            if crash.get(dt, False):
                label[dt] = CRASH
                reason[dt] = (f"vol pctile {f['vol_pctile'].get(dt, np.nan):.0%} ≥ "
                              f"{r['crash_vol_pctile']:.0%} or 1m DD "
                              f"{f['dd_1m'].get(dt, np.nan):.1%} ≤ {r['crash_drawdown']:.0%}")
            elif up.get(dt, False):
                label[dt] = TREND_UP
                reason[dt] = (f"fastMA>slowMA and ADX {f['adx'].get(dt, np.nan):.0f} "
                              f"≥ {r['adx_trend_min']:.0f}")
            elif down.get(dt, False):
                label[dt] = TREND_DOWN
                reason[dt] = (f"fastMA<slowMA and ADX {f['adx'].get(dt, np.nan):.0f} "
                              f"≥ {r['adx_trend_min']:.0f}")
            elif rng.get(dt, False):
                label[dt] = RANGE
                reason[dt] = f"ADX {f['adx'].get(dt, np.nan):.0f} ≤ {r['adx_range_max']:.0f} (chop)"
            else:
                # transitional zone (range_max < ADX < trend_min): treat as a
                # WEAK trend in the direction of the MA stack rather than dumping
                # it into RANGE, so mean-reversion only fires in genuine chop.
                if f["ma_fast"].get(dt, np.nan) >= f["ma_slow"].get(dt, np.nan):
                    label[dt] = TREND_UP
                    reason[dt] = (f"weak uptrend: ADX {f['adx'].get(dt, np.nan):.0f} "
                                  f"in [{r['adx_range_max']:.0f},{r['adx_trend_min']:.0f}], "
                                  f"fastMA>slowMA")
                else:
                    label[dt] = TREND_DOWN
                    reason[dt] = (f"weak downtrend: ADX {f['adx'].get(dt, np.nan):.0f} "
                                  f"in [{r['adx_range_max']:.0f},{r['adx_trend_min']:.0f}], "
                                  f"fastMA<slowMA")
        return pd.DataFrame({"label": label, "reason": reason})


# =====================================================================
# 2. Volatility-clustering regime (EWMA vol state; GARCH-like proxy).
# =====================================================================
class VolRegime:
    name = "vol_cluster"

    def __init__(self, cfg: dict):
        self.r = cfg["regime"]

    def classify(self, f: pd.DataFrame) -> pd.DataFrame:
        # EWMA conditional vol (RiskMetrics lambda=0.94) — the classic GARCH proxy.
        ret = f["ret"].fillna(0.0)
        ewma_var = ret.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean()
        cvol = np.sqrt(ewma_var) * np.sqrt(self.r["vol_window"]) * np.sqrt(252 / self.r["vol_window"])
        pct = ind.rolling_percentile(cvol, self.r["vol_lookback_pctile"])
        slope = f["close"].pct_change(self.r["ma_fast"])

        label = pd.Series(index=f.index, dtype="object")
        reason = pd.Series(index=f.index, dtype="object")
        for dt in f.index:
            p = pct.get(dt, np.nan)
            s = slope.get(dt, np.nan)
            if np.isnan(p):
                label[dt], reason[dt] = RANGE, "insufficient history"
            elif p >= self.r["crash_vol_pctile"]:
                label[dt] = CRASH
                reason[dt] = f"EWMA cond. vol in {p:.0%} pctile (cluster of large moves)"
            elif p <= 0.40:
                label[dt] = RANGE
                reason[dt] = f"low vol cluster (pctile {p:.0%}) -> compression/range"
            else:
                label[dt] = TREND_UP if s > 0 else TREND_DOWN
                reason[dt] = (f"mid vol (pctile {p:.0%}), {self.r['ma_fast']}d slope "
                              f"{s:+.1%} -> directional")
        return pd.DataFrame({"label": label, "reason": reason})


# =====================================================================
# 3. Hidden Markov Model regime (analytical overlay).
#    NOTE on lookahead: 'analytical' mode fits on the full sample, so it is
#    NOT causal and must not drive live fills. It is for labelling/research.
#    Set regime.hmm_mode='walkforward' for a causal (expanding-refit) variant.
# =====================================================================
class HMMRegime:
    name = "hmm"

    def __init__(self, cfg: dict):
        self.r = cfg["regime"]
        self.n = cfg["regime"]["hmm_states"]
        self.mode = cfg["regime"]["hmm_mode"]
        self.refit = cfg["regime"]["hmm_walkforward_refit"]

    def _features(self, f: pd.DataFrame) -> np.ndarray:
        x = np.column_stack([
            f["ret"].fillna(0.0).values,
            f["ret"].abs().fillna(0.0).values,        # |ret| ~ vol clustering
            f["atr_pct"].ffill().fillna(0.0).values,
        ])
        return x

    def _fit_predict(self, X: np.ndarray):
        try:
            import warnings
            from hmmlearn.hmm import GaussianHMM
            Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            mu, sd = Xc.mean(0), Xc.std(0)
            sd[sd == 0] = 1.0
            Xs = (Xc - mu) / sd
            model = GaussianHMM(n_components=self.n, covariance_type="diag",
                                n_iter=100, random_state=42, min_covar=1e-3,
                                init_params="stmc")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xs)
                states = model.predict(Xs)
            if not np.isfinite(model.startprob_).all():
                return None, None
            return model, states
        except Exception:  # noqa: BLE001
            return None, None

    def _map_states(self, model, f: pd.DataFrame, states: np.ndarray) -> dict:
        """Map raw HMM state ids -> economic regimes by their mean ret & vol."""
        df = pd.DataFrame({"state": states, "ret": f["ret"].values,
                           "vol": f["ret"].abs().values}, index=f.index)
        stats = df.groupby("state").agg(mu=("ret", "mean"), sd=("vol", "mean"))
        mapping = {}
        crash_state = stats["sd"].idxmax()
        mapping[crash_state] = CRASH
        rest = stats.drop(index=crash_state)
        if len(rest):
            range_state = rest["sd"].idxmin()
            mapping[range_state] = RANGE
            rest2 = rest.drop(index=range_state)
            for st in rest2.index:
                mapping[st] = TREND_UP if rest2.loc[st, "mu"] >= 0 else TREND_DOWN
        return mapping

    def classify(self, f: pd.DataFrame) -> pd.DataFrame:
        X = self._features(f)
        label = pd.Series(index=f.index, dtype="object")
        reason = pd.Series(index=f.index, dtype="object")

        if self.mode == "walkforward":
            # Causal: at each refit point, fit on data so far, label forward block.
            i = max(252, self.n * 30)
            label[:] = RANGE
            while i < len(f):
                model, states = self._fit_predict(X[:i])
                if model is None:
                    reason[:] = "hmmlearn unavailable"
                    label[:] = RANGE
                    return pd.DataFrame({"label": label, "reason": reason})
                mp = self._map_states(model, f.iloc[:i], states)
                j = min(i + self.refit, len(f))
                fwd = model.predict(X[:j])[i:j]
                for k, st in zip(range(i, j), fwd):
                    label.iloc[k] = mp.get(st, RANGE)
                    reason.iloc[k] = f"HMM(walk-fwd) state {st}"
                i = j
            return pd.DataFrame({"label": label, "reason": reason})

        # analytical (full-sample) mode
        model, states = self._fit_predict(X)
        if model is None:
            label[:] = RANGE
            reason[:] = "hmmlearn unavailable -> RANGE"
            return pd.DataFrame({"label": label, "reason": reason})
        mp = self._map_states(model, f, states)
        for dt, st in zip(f.index, states):
            label[dt] = mp.get(st, RANGE)
            reason[dt] = f"HMM state {st} (mean-ret/vol mapped)"
        return pd.DataFrame({"label": label, "reason": reason})


# =====================================================================
# Ensemble — combines detectors with a documented priority + vote.
# =====================================================================
class RegimeEnsemble:
    """
    Decision logic (transparent on purpose):
      * CRASH is a *veto*: if the causal rule engine flags crash, crash wins.
        (Capital preservation must never be out-voted.)
      * Otherwise: majority vote across detectors. Ties -> the causal rule
        engine breaks them (it is the only fully no-lookahead member).
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.detectors = [RuleRegime(cfg), VolRegime(cfg)]
        if cfg["regime"].get("hmm_enabled", True):
            self.detectors.append(HMMRegime(cfg))
        self._per_detector: dict[str, pd.DataFrame] = {}
        self.features: pd.DataFrame | None = None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.features = build_features(df, self.cfg)
        f = self.features
        outs = {d.name: d.classify(f) for d in self.detectors}
        self._per_detector = outs

        rule = outs["rule_ma_atr"]["label"]
        votes = pd.DataFrame({name: o["label"] for name, o in outs.items()})

        final = pd.Series(index=f.index, dtype="object")
        why = pd.Series(index=f.index, dtype="object")
        for dt in f.index:
            row = votes.loc[dt].dropna()
            if rule.get(dt) == CRASH:
                final[dt] = CRASH
                why[dt] = "CRASH veto from causal rule engine (capital preservation)"
                continue
            if row.empty:
                final[dt], why[dt] = RANGE, "no detector ready"
                continue
            counts = row.value_counts()
            top = counts.index[0]
            if (counts == counts.iloc[0]).sum() > 1:  # tie
                top = rule.get(dt, RANGE)
                why[dt] = f"tie {dict(counts)} -> rule tiebreak {top}"
            else:
                why[dt] = f"majority {dict(counts)}"
            final[dt] = top

        out = pd.DataFrame({"regime": final, "why": why})
        for name, o in outs.items():
            out[f"d_{name}"] = o["label"]
        out["reason_rule"] = outs["rule_ma_atr"]["reason"]
        return out

    # ---- explainability API -------------------------------------------
    def explain(self, regimes: pd.DataFrame, date) -> str:
        date = pd.Timestamp(date)
        if date not in regimes.index:
            date = regimes.index[regimes.index <= date][-1]
        row = regimes.loc[date]
        f = self.features.loc[date]
        lines = [
            f"=== Regime explanation @ {date.date()} ===",
            f"FINAL REGIME : {row['regime']}",
            f"DECISION     : {row['why']}",
            "FEATURES     : "
            f"ADX={f['adx']:.0f}  fastMA{'>' if f['ma_fast']>f['ma_slow'] else '<'}slowMA  "
            f"ATR%={f['atr_pct']:.2%}  rVol={f['rvol']:.1%}  "
            f"volPctile={f['vol_pctile']:.0%}  1mDD={f['dd_1m']:.1%}",
            "DETECTORS    :",
        ]
        for name, o in self._per_detector.items():
            lines.append(f"   - {name:<12}: {o.loc[date,'label']:<14} ({o.loc[date,'reason']})")
        return "\n".join(lines)
