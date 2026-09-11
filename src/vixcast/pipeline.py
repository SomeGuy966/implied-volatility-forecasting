"""End-to-end run: data -> walk-forward forecasts -> metrics -> artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import evaluation as ev
from .backtest import walk_forward
from .config import Config
from .data import load_close, to_log
from .models import build_models
from .report import summary_markdown, write_frames

log = logging.getLogger(__name__)

BENCHMARK = "rw"


@dataclass(frozen=True)
class RunOutputs:
    forecasts: pd.DataFrame
    point: pd.DataFrame
    intervals: pd.DataFrame
    distribution: pd.DataFrame
    calibration: pd.DataFrame
    summary: str
    files: tuple[Path, ...]


def run(cfg: Config) -> RunOutputs:
    close = load_close(cfg.ticker, cfg.data_start, cfg.cache_path, refresh=cfg.refresh_data)
    log_close = to_log(close)
    log.info(
        "Series spans %s → %s (%d trading days)",
        close.index[0].date(),
        close.index[-1].date(),
        len(close),
    )

    keys = list(cfg.models)
    if BENCHMARK not in keys:  # the benchmark is always evaluated so ratios are defined
        keys.insert(0, BENCHMARK)
    models = build_models(keys, seed=cfg.seed)
    labels = {m.key: m.label for m in models}

    fc = walk_forward(log_close, models, cfg)

    point = ev.point_metrics(fc, keys, BENCHMARK)
    intervals = ev.interval_metrics(fc, keys, cfg.levels, BENCHMARK)
    distribution = ev.distribution_metrics(fc, keys, cfg.quantiles, BENCHMARK)
    calibration = ev.calibration_curve(fc, keys, cfg.quantiles)

    files = list(
        write_frames(
            {
                "forecasts": fc,
                "metrics_point": point,
                "metrics_intervals": intervals,
                "metrics_distribution": distribution,
                "calibration": calibration,
            },
            cfg.results_dir,
        )
    )
    summary = summary_markdown(
        cfg,
        point,
        intervals,
        distribution,
        labels,
        fc["target_date"].min(),
        fc["target_date"].max(),
    )
    summary_path = cfg.results_dir / "summary.md"
    summary_path.write_text(summary)
    files.append(summary_path)

    if cfg.make_plots:
        from . import plotting

        # EWMA is a deliberately weak baseline; it swamps the axes of every
        # comparative figure, so it only appears in the tables and the overlay.
        serious = [k for k in keys if k != "ewma"]
        files += [
            plotting.plot_forecast_overlay(
                fc, keys, labels, cfg.figures_dir / "forecast_overlay.png"
            ),
            plotting.plot_prediction_bands(
                fc, serious, labels, 0.90, cfg.figures_dir / "prediction_bands_90.png"
            ),
            plotting.plot_calibration(
                calibration, labels, cfg.figures_dir / "calibration.png", n=int(point["n"].iloc[0])
            ),
            plotting.plot_relative_performance(
                ev.cumulative_loss_difference(fc, serious, BENCHMARK),
                ev.rolling_rmse_ratio(fc, serious, 63, BENCHMARK),
                labels,
                labels[BENCHMARK],
                cfg.figures_dir / "relative_performance.png",
            ),
        ]

    return RunOutputs(fc, point, intervals, distribution, calibration, summary, tuple(files))
