"""Static figures for the README and for eyeballing a run.

All figures share one palette and style so they read as a set. The realised VIX is
always near-black; models take categorical hues in a fixed, colour-blind-safe order
(the assignment follows the model, never its rank in a table).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # headless; the CLI never opens windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .backtest import band_cols

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4e0"
ACTUAL = "#1a1a19"

# Fixed slot per model key; the order was validated for adjacent-pair CVD separation.
SERIES = {
    "rw": "#2a78d6",  # blue
    "ewma": "#eb6834",  # orange
    "har": "#1baf7a",  # aqua
    "ar_garch": "#eda100",  # yellow
    "har_garch": "#e87ba4",  # magenta
}
_EXTRA = ["#008300", "#4a3aa7", "#e34948"]  # spare slots for user-added models

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.labelsize": 9,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "font.family": "sans-serif",
        "text.color": INK,
    }
)


def _color(key: str, i: int) -> str:
    return SERIES.get(key, _EXTRA[i % len(_EXTRA)])


def _wide(fc: pd.DataFrame, col: str, models: Sequence[str]) -> pd.DataFrame:
    sub = fc[fc["model"].isin(models)]
    return sub.pivot(index="target_date", columns="model", values=col).dropna()


def _actual(fc: pd.DataFrame) -> pd.Series:
    first = fc["model"].iloc[0]
    actual: pd.Series = fc[fc["model"] == first].set_index("target_date")["y_true"]
    return actual


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------- figures


def plot_forecast_overlay(
    fc: pd.DataFrame, models: Sequence[str], labels: dict[str, str], path: Path, zoom_days: int = 60
) -> Path:
    """Realised VIX against every point forecast: whole test window, plus a zoom on
    the largest spike, where one-day-ahead differences are actually visible."""
    points = _wide(fc, "point", models)
    actual = _actual(fc).reindex(points.index)

    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1.15, 1]}
    )
    for ax in (ax_full, ax_zoom):
        ax.plot(actual.index, actual, color=ACTUAL, lw=1.4, label="Realised VIX", zorder=5)
        for i, key in enumerate(models):
            ax.plot(
                points.index,
                points[key],
                color=_color(key, i),
                lw=1.3,
                alpha=0.95,
                label=labels[key],
            )
        ax.set_ylabel("VIX")

    peak = actual.idxmax()
    pos = int(np.flatnonzero(points.index == peak)[0])
    lo, hi = max(0, pos - zoom_days // 3), min(len(points) - 1, pos + 2 * zoom_days // 3)
    ax_zoom.set_xlim(points.index[lo], points.index[hi])
    window = actual.iloc[lo : hi + 1]
    ax_zoom.set_ylim(window.min() * 0.9, window.max() * 1.05)
    ax_full.axvspan(points.index[lo], points.index[hi], color=GRID, alpha=0.6, zorder=0)

    h = int(fc["horizon"].iloc[0])
    ax_full.set_title(f"{h}-day-ahead point forecasts over the test window")
    ax_zoom.set_title(f"Zoom: {zoom_days} trading days around the largest spike")
    ax_full.legend(ncol=len(models) + 1, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_prediction_bands(
    fc: pd.DataFrame, models: Sequence[str], labels: dict[str, str], level: float, path: Path
) -> Path:
    """One panel per model: realised VIX, the central band, and misses marked."""
    lo_col, hi_col = band_cols(level)
    lo_w, hi_w = _wide(fc, lo_col, models), _wide(fc, hi_col, models)
    actual = _actual(fc).reindex(lo_w.index)

    fig, axes = plt.subplots(
        len(models), 1, figsize=(11, 2.6 * len(models)), sharex=True, sharey=True
    )
    axes_list: list[Axes] = list(np.atleast_1d(axes))
    for i, (key, ax) in enumerate(zip(models, axes_list, strict=True)):
        c = _color(key, i)
        lo, hi = lo_w[key], hi_w[key]
        ax.fill_between(lo.index, lo, hi, color=c, alpha=0.22, lw=0, label=f"{level:.0%} interval")
        ax.plot(lo.index, lo, color=c, lw=0.8, alpha=0.8)
        ax.plot(hi.index, hi, color=c, lw=0.8, alpha=0.8)
        ax.plot(actual.index, actual, color=ACTUAL, lw=1.1, label="Realised VIX")
        miss = (actual < lo) | (actual > hi)
        ax.scatter(
            actual.index[miss],
            actual[miss],
            s=14,
            color="#e34948",
            zorder=6,
            label=f"miss ({miss.mean():.1%})",
        )
        ax.set_title(labels[key], loc="left")
        ax.set_ylabel("VIX")
        ax.legend(loc="upper left", ncol=3)
    h = int(fc["horizon"].iloc[0])
    fig.suptitle(
        f"{level:.0%} prediction intervals, {h}-day-ahead",
        y=1.0,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(curve: pd.DataFrame, labels: dict[str, str], path: Path, n: int) -> Path:
    """Empirical minus nominal coverage of central intervals, with the ±2 s.e. band a
    perfectly calibrated model would stay inside given ``n`` test observations."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    grid = np.linspace(0.05, 0.99, 200)
    se = np.sqrt(grid * (1 - grid) / n)
    ax.fill_between(
        grid, -1.96 * se, 1.96 * se, color=GRID, alpha=0.8, lw=0, label="±1.96 s.e. (binomial)"
    )
    ax.axhline(0, color=INK_SOFT, lw=1)
    for i, (key, sub) in enumerate(curve.groupby("model", sort=False)):
        sub = sub.sort_values("nominal")
        ax.plot(
            sub["nominal"],
            sub["empirical"] - sub["nominal"],
            color=_color(str(key), i),
            marker="o",
            ms=4,
            label=labels[str(key)],
        )
    ax.set_xlim(0.05, 1.0)
    ax.set_xlabel("Nominal coverage of central interval")
    ax.set_ylabel("Empirical − nominal coverage")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Unconditional interval calibration")
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    return _save(fig, path)


def plot_relative_performance(
    cum_diff: pd.DataFrame,
    rolling_ratio: pd.DataFrame,
    labels: dict[str, str],
    benchmark_label: str,
    path: Path,
    window: int = 63,
) -> Path:
    """Top: cumulative squared-error advantage over the benchmark (up = model winning).
    Bottom: rolling RMSE ratio to the benchmark (below 1 = model winning)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for i, key in enumerate(cum_diff.columns):
        ax1.plot(cum_diff.index, cum_diff[key], color=_color(str(key), i), label=labels[str(key)])
    ax1.axhline(0, color=INK_SOFT, lw=1)
    ax1.set_title(
        f"Cumulative squared-error advantage over {benchmark_label} (above 0 = model ahead)"
    )
    ax1.set_ylabel("Σ (SE benchmark − SE model)")
    ax1.legend(loc="upper left", ncol=len(cum_diff.columns))

    for i, key in enumerate(rolling_ratio.columns):
        ax2.plot(
            rolling_ratio.index,
            rolling_ratio[key],
            color=_color(str(key), i),
            label=labels[str(key)],
        )
    ax2.axhline(1, color=INK_SOFT, lw=1)
    ax2.set_title(
        f"Rolling {window}-day RMSE relative to {benchmark_label} (below 1 = model better)"
    )
    ax2.set_ylabel("RMSE ratio")
    ax2.set_ylim(0.75, 1.25)
    fig.tight_layout()
    return _save(fig, path)
