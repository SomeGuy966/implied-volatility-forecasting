"""Common forecaster interface.

Every model sees the *log* VIX history up to and including the forecast origin and
returns a distributional forecast of the log VIX ``horizon`` trading days later. The
walk-forward engine converts log-space output to the level scale, so models never
touch the raw index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LogForecast:
    """Point and quantile forecast of log VIX at ``origin + horizon``.

    ``quantiles[i]`` is the forecast quantile at probability ``probs[i]``. The point
    forecast is the conditional mean of log VIX; exponentiating it gives a *median*
    forecast on the level scale, which is what MAE rewards.
    """

    point: float
    probs: tuple[float, ...]
    quantiles: FloatArray
    converged: bool = True

    def __post_init__(self) -> None:
        if len(self.probs) != len(self.quantiles):
            raise ValueError("probs and quantiles must have the same length")
        if not np.all(np.isfinite(self.quantiles)) or not np.isfinite(self.point):
            raise ValueError("forecast contains non-finite values")


class Forecaster(ABC):
    """A one-shot forecaster: fit on history, forecast ``horizon`` steps ahead.

    Implementations must be picklable (they are shipped to joblib workers) and must
    not carry state between calls.
    """

    key: str  # short identifier used in CSV columns / CLI flags
    label: str  # human-readable name used in tables and plots
    expensive: bool = False  # True => the backtest farms fits out to worker processes

    @abstractmethod
    def forecast(
        self, log_history: FloatArray, horizon: int, probs: Sequence[float]
    ) -> LogForecast:
        """Forecast log VIX at ``t + horizon`` given ``log_history[..t]``."""

    def min_history(self, horizon: int) -> int:
        """Smallest history length the model can be fitted on."""
        return 30 + horizon


def empirical_quantiles(residuals: FloatArray, probs: Sequence[float]) -> FloatArray:
    """Quantiles of a residual sample, used for non-parametric interval forecasts."""
    if residuals.size < 10:
        raise ValueError("need at least 10 residuals for empirical quantiles")
    return np.asarray(np.quantile(residuals, np.asarray(probs, dtype=float)), dtype=float)


def h_step_targets(x: FloatArray, horizon: int) -> tuple[FloatArray, FloatArray]:
    """Pair each observation with its value ``horizon`` steps later.

    Returns ``(x_t, x_{t+h})`` arrays of equal length, aligned so that row ``i`` is a
    valid (feature, target) pair with no look-ahead.
    """
    if horizon >= len(x):
        raise ValueError("horizon must be shorter than the history")
    return x[:-horizon], x[horizon:]
