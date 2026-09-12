"""Naive baselines every serious model has to beat.

Both baselines produce intervals from the empirical distribution of their own past
``h``-step errors, so they are legitimate distributional forecasters and the interval
metrics compare like with like.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .base import FloatArray, Forecaster, LogForecast, empirical_quantiles, h_step_targets


class RandomWalk(Forecaster):
    """Tomorrow's log VIX is today's. The benchmark for point *and* interval accuracy."""

    key = "rw"
    label = "Random walk"

    def forecast(
        self, log_history: FloatArray, horizon: int, probs: Sequence[float]
    ) -> LogForecast:
        x_t, x_th = h_step_targets(log_history, horizon)
        residuals = x_th - x_t
        point = float(log_history[-1])
        return LogForecast(point, tuple(probs), point + empirical_quantiles(residuals, probs))


class EWMA(Forecaster):
    """Exponentially weighted mean of log VIX as the forecast.

    A smoother, not a forecaster: it lags a persistent series by construction. Kept as
    a sanity check that the evaluation harness punishes lag.
    """

    key = "ewma"
    label = "EWMA"

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        self.alpha = alpha

    def forecast(
        self, log_history: FloatArray, horizon: int, probs: Sequence[float]
    ) -> LogForecast:
        smoothed = pd.Series(log_history).ewm(alpha=self.alpha, adjust=False).mean().to_numpy()
        # residual of the smoothed value at t against the realised value at t + h
        e_t, x_th = smoothed[:-horizon], log_history[horizon:]
        residuals = x_th - e_t
        point = float(smoothed[-1])
        return LogForecast(point, tuple(probs), point + empirical_quantiles(residuals, probs))
