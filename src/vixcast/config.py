"""Run configuration.

Everything that controls a backtest lives in one frozen dataclass so a run can be
described (and reproduced) by a single object. The CLI is a thin layer that builds
one of these from flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODELS: tuple[str, ...] = ("rw", "ewma", "har", "ar_garch", "har_garch")
DEFAULT_LEVELS: tuple[float, ...] = (0.80, 0.90, 0.95)

# Dense grid used for calibration curves. The band-specific quantiles (e.g. 0.025 /
# 0.975 for a 95% band) are added on top of this by ``Config.quantiles``.
_CALIBRATION_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))


@dataclass(frozen=True)
class Config:
    """Immutable description of one backtest run."""

    ticker: str = "^VIX"
    data_start: str = "2010-01-01"
    test_start: str = "2024-01-01"
    test_end: str | None = None
    horizon: int = 1
    window: int | None = None  # rolling estimation window in trading days; None = expanding
    min_train: int = 260  # smallest history (obs) a model may be fitted on
    levels: tuple[float, ...] = DEFAULT_LEVELS
    models: tuple[str, ...] = DEFAULT_MODELS
    n_jobs: int = -1
    seed: int = 0
    refresh_data: bool = False
    make_plots: bool = True
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    figures_dir: Path = field(default_factory=lambda: Path("figures"))

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.min_train < 30:
            raise ValueError("min_train must be >= 30")
        if self.window is not None and self.window < self.min_train:
            raise ValueError("window must be >= min_train")
        for lvl in self.levels:
            if not 0.0 < lvl < 1.0:
                raise ValueError(f"interval level {lvl} must lie in (0, 1)")
        if not self.models:
            raise ValueError("at least one model is required")

    @property
    def quantiles(self) -> tuple[float, ...]:
        """Sorted probability levels every model must return a quantile for."""
        qs: set[float] = set(_CALIBRATION_GRID) | {0.5}
        for lvl in self.levels:
            a = 1.0 - lvl
            qs.add(round(a / 2, 4))
            qs.add(round(1.0 - a / 2, 4))
        return tuple(sorted(qs))

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "vix.csv"
