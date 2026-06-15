"""
Experiment Tracking  (Priority #5)
=================================
Thin wrapper around MLflow so each pipeline run logs its config parameters and
headline metrics (Sharpe, drawdown, profit factor, walk-forward results...).
If mlflow isn't installed, everything degrades to a local JSON log so the
pipeline never breaks.

    from src.analytics.experiment_tracking import ExperimentTracker
    with ExperimentTracker(cfg) as tr:
        tr.log_params({...}); tr.log_metrics({...})
"""
from __future__ import annotations
import json
import time
from pathlib import Path


class ExperimentTracker:
    def __init__(self, cfg: dict, run_name: str | None = None):
        self.cfg = cfg
        self.run_name = run_name or f"run_{int(time.time())}"
        self.out = Path(cfg["report"]["out_dir"]); self.out.mkdir(parents=True, exist_ok=True)
        self._mlflow = None
        self._payload = {"run": self.run_name, "params": {}, "metrics": {}}
        try:
            import mlflow
            mlflow.set_experiment("market_regime_engine")
            self._mlflow = mlflow
            self._active = mlflow.start_run(run_name=self.run_name)
        except Exception:                          # mlflow missing or no server
            self._mlflow = None

    def log_params(self, params: dict):
        flat = _flatten(params)
        self._payload["params"].update(flat)
        if self._mlflow:
            for k, v in flat.items():
                try: self._mlflow.log_param(k[:250], v)
                except Exception: pass

    def log_metrics(self, metrics: dict):
        clean = {k: float(v) for k, v in _flatten(metrics).items()
                 if _is_num(v)}
        self._payload["metrics"].update(clean)
        if self._mlflow:
            for k, v in clean.items():
                try: self._mlflow.log_metric(k[:250], v)
                except Exception: pass

    def log_artifact(self, path: str):
        if self._mlflow:
            try: self._mlflow.log_artifact(path)
            except Exception: pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        (self.out / "experiment_log.json").write_text(json.dumps(self._payload, indent=2))
        if self._mlflow:
            try: self._mlflow.end_run()
            except Exception: pass


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)
