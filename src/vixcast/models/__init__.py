"""Model registry.

Add a model by implementing :class:`Forecaster` and registering a factory here; the
CLI, backtest and report pick it up by key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .base import Forecaster, LogForecast
from .baselines import EWMA, RandomWalk
from .garch import ArchGarch, ar_garch, har_garch
from .har import HAR

_REGISTRY: dict[str, Callable[[int], Forecaster]] = {
    "rw": lambda _seed: RandomWalk(),
    "ewma": lambda _seed: EWMA(alpha=0.05),
    "har": lambda _seed: HAR(),
    "ar_garch": ar_garch,
    "har_garch": har_garch,
}

AVAILABLE_MODELS: tuple[str, ...] = tuple(_REGISTRY)


def build_models(keys: Sequence[str], seed: int = 0) -> list[Forecaster]:
    unknown = [k for k in keys if k not in _REGISTRY]
    if unknown:
        raise KeyError(f"unknown model(s) {unknown}; available: {AVAILABLE_MODELS}")
    return [_REGISTRY[k](seed) for k in keys]


__all__ = [
    "AVAILABLE_MODELS",
    "EWMA",
    "HAR",
    "ArchGarch",
    "Forecaster",
    "LogForecast",
    "RandomWalk",
    "ar_garch",
    "build_models",
    "har_garch",
]
