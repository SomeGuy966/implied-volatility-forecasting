"""Heterogeneous Autoregression (Corsi, 2009) on log VIX.

The HAR regresses the target on the last value and on the 5- and 22-day averages of
the series. It is the workhorse benchmark of the realised-volatility literature
because it captures slow mean reversion with three parameters and is estimated by
OLS in microseconds. Here the regression is *direct* for the horizon: ``x_{t+h}`` is
regressed on features at ``t``, so no iteration of one-step forecasts is needed.

Intervals come from the empirical quantiles of in-sample residuals - a deliberately
non-parametric, homoskedastic choice that contrasts with the GARCH models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import FloatArray, Forecaster, LogForecast, empirical_quantiles


def har_design(x: FloatArray, lags: Sequence[int]) -> FloatArray:
    """Design matrix with an intercept and trailing means over each lag length.

    Row ``i`` holds the features at ``t = max(lags) - 1 + i``; the last row is the
    feature vector at the final observation.
    """
    n = len(x)
    max_lag = max(lags)
    if n < max_lag:
        raise ValueError("history shorter than the longest HAR lag")
    csum = np.concatenate(([0.0], np.cumsum(x)))
    cols = []
    for lag in lags:
        window_mean = (csum[lag:] - csum[:-lag]) / lag  # windows ending at t = lag-1 .. n-1
        cols.append(window_mean[max_lag - lag :])
    return np.column_stack([np.ones(n - max_lag + 1), *cols])


class HAR(Forecaster):
    key = "har"
    label = "HAR"

    def __init__(self, lags: Sequence[int] = (1, 5, 22)) -> None:
        if any(lag < 1 for lag in lags) or len(set(lags)) != len(lags):
            raise ValueError("lags must be distinct positive integers")
        self.lags = tuple(sorted(lags))

    def min_history(self, horizon: int) -> int:
        return max(self.lags) + horizon + 30

    def forecast(
        self, log_history: FloatArray, horizon: int, probs: Sequence[float]
    ) -> LogForecast:
        X = har_design(log_history, self.lags)
        y = log_history[max(self.lags) - 1 + horizon :]  # x_{t+h} aligned with X rows
        X_train = X[:-horizon]
        beta, *_ = np.linalg.lstsq(X_train, y, rcond=None)
        residuals = y - X_train @ beta
        point = float(X[-1] @ beta)
        return LogForecast(point, tuple(probs), point + empirical_quantiles(residuals, probs))
