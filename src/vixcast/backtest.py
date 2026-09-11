"""Walk-forward (rolling-origin) backtest.

For every trading day ``d`` in the test period and every model, the model is fitted
on the log-VIX history up to the *origin* ``d - h`` trading days (positionally, not
calendar days, so weekends and holidays never leak in) and asked for a distributional
forecast of log VIX at ``d``. Forecasts are exponentiated to the VIX level and stored
in a long-format frame with one row per (model, target date).

The engine is the only place that touches the date index; models see plain arrays.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .models.base import FloatArray, Forecaster, LogForecast

log = logging.getLogger(__name__)


def quantile_col(p: float) -> str:
    return f"q{p:g}"


def quantile_cols(probs: Sequence[float]) -> list[str]:
    return [quantile_col(p) for p in probs]


def _forecast_one(
    model: Forecaster, history: FloatArray, horizon: int, probs: Sequence[float]
) -> LogForecast:
    return model.forecast(history, horizon, probs)


def walk_forward(log_level: pd.Series, models: Sequence[Forecaster], cfg: Config) -> pd.DataFrame:
    """Run every model through the test period and return level-scale forecasts."""
    x = log_level.to_numpy(dtype=float)
    idx = pd.DatetimeIndex(log_level.index)
    h = cfg.horizon
    probs = cfg.quantiles

    test_mask = idx >= pd.Timestamp(cfg.test_start)
    if cfg.test_end is not None:
        test_mask &= idx <= pd.Timestamp(cfg.test_end)
    target_positions = np.flatnonzero(test_mask)
    if target_positions.size == 0:
        raise ValueError("no test observations in the requested window")

    frames: list[pd.DataFrame] = []
    for model in models:
        need = max(cfg.min_train, model.min_history(h))
        tasks: list[tuple[int, int, int]] = []
        for tp in map(int, target_positions):
            op = tp - h  # origin position
            start = 0 if cfg.window is None else max(0, op + 1 - cfg.window)
            if op + 1 - start < need:
                continue
            tasks.append((tp, op, start))
        if not tasks:
            raise ValueError(f"{model.key}: no test date has enough history")

        n_jobs = cfg.n_jobs if model.expensive else 1
        t0 = time.perf_counter()
        results: list[LogForecast] = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_forecast_one)(model, x[start : op + 1], h, probs) for _, op, start in tasks
        )
        elapsed = time.perf_counter() - t0
        n_fail = sum(not r.converged for r in results)
        log.info(
            "%-10s %4d forecasts in %6.1fs (%5.1f ms each)%s",
            model.key,
            len(results),
            elapsed,
            1e3 * elapsed / len(results),
            f"  [{n_fail} non-converged]" if n_fail else "",
        )

        frame = pd.DataFrame(
            {
                "target_date": idx[[tp for tp, _, _ in tasks]],
                "origin_date": idx[[op for _, op, _ in tasks]],
                "model": model.key,
                "horizon": h,
                "y_true": np.exp(x[[tp for tp, _, _ in tasks]]),
                "point": np.exp([r.point for r in results]),
                "converged": [r.converged for r in results],
            }
        )
        q_matrix = np.exp(np.vstack([r.quantiles for r in results]))
        frame[quantile_cols(probs)] = q_matrix
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    return out


def band_cols(level: float) -> tuple[str, str]:
    """Column names for the lower/upper quantile of a central ``level`` band."""
    a = 1.0 - level
    return quantile_col(round(a / 2, 4)), quantile_col(round(1.0 - a / 2, 4))
