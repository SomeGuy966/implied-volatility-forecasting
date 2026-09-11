"""Persist run outputs and render a Markdown summary of the metric tables."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .config import Config


def _fmt(value: object, spec: str) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "–"
    if isinstance(value, float | int) and not isinstance(value, bool):
        return format(value, spec)
    return str(value)


def to_markdown(df: pd.DataFrame, formats: Mapping[str, str], headers: Mapping[str, str]) -> str:
    """Render selected columns of ``df`` as a GitHub-flavoured Markdown table.

    ``formats`` maps column -> format spec (also fixes the column order); ``headers``
    maps column -> display name. Written by hand to avoid a ``tabulate`` dependency.
    """
    cols = list(formats)
    head = "| " + " | ".join(headers.get(c, c) for c in cols) + " |"
    align = "| " + " | ".join(":--" if c == cols[0] else "--:" for c in cols) + " |"
    body = [
        "| " + " | ".join(_fmt(row[c], formats[c]) for c in cols) + " |" for _, row in df.iterrows()
    ]
    return "\n".join([head, align, *body])


def _stars(p: float) -> str:
    if math.isnan(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _with_labels(df: pd.DataFrame, labels: Mapping[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].map(lambda k: labels.get(k, k))
    return out


def point_table(point: pd.DataFrame, labels: Mapping[str, str]) -> str:
    df = _with_labels(point, labels)
    df["dm_se"] = [
        f"{s:+.2f}{_stars(p)}" if pd.notna(s) else "–"
        for s, p in zip(df["dm_stat_se"], df["dm_p_se"], strict=True)
    ]
    df["dm_ae"] = [
        f"{s:+.2f}{_stars(p)}" if pd.notna(s) else "–"
        for s, p in zip(df["dm_stat_ae"], df["dm_p_ae"], strict=True)
    ]
    return to_markdown(
        df,
        {
            "model": "s",
            "rmse": ".3f",
            "mae": ".3f",
            "rmse_ratio": ".3f",
            "dm_se": "s",
            "dm_ae": "s",
        },
        {
            "model": "Model",
            "rmse": "RMSE",
            "mae": "MAE",
            "rmse_ratio": "RMSE / RW",
            "dm_se": "DM (sq. loss)",
            "dm_ae": "DM (abs. loss)",
        },
    )


def interval_table(intervals: pd.DataFrame, labels: Mapping[str, str], level: float) -> str:
    df = _with_labels(intervals[intervals["level"] == level], labels)
    df["dm_is"] = [
        f"{s:+.2f}{_stars(p)}" if pd.notna(s) else "–"
        for s, p in zip(df["dm_stat_is"], df["dm_p_is"], strict=True)
    ]
    return to_markdown(
        df,
        {
            "model": "s",
            "coverage": ".1%",
            "avg_width": ".2f",
            "interval_score": ".2f",
            "is_ratio": ".3f",
            "kupiec_p": ".2f",
            "christoffersen_p": ".2f",
            "dm_is": "s",
        },
        {
            "model": "Model",
            "coverage": "Coverage",
            "avg_width": "Avg width",
            "interval_score": "Interval score",
            "is_ratio": "IS / RW",
            "kupiec_p": "Kupiec p",
            "christoffersen_p": "Christoffersen p",
            "dm_is": "DM (IS)",
        },
    )


def distribution_table(dist: pd.DataFrame, labels: Mapping[str, str]) -> str:
    df = _with_labels(dist, labels)
    df["dm"] = [
        f"{s:+.2f}{_stars(p)}" if pd.notna(s) else "–"
        for s, p in zip(df["dm_stat"], df["dm_p"], strict=True)
    ]
    return to_markdown(
        df,
        {"model": "s", "pinball": ".4f", "pinball_ratio": ".3f", "dm": "s"},
        {"model": "Model", "pinball": "Avg pinball loss", "pinball_ratio": "vs RW", "dm": "DM"},
    )


def summary_markdown(
    cfg: Config,
    point: pd.DataFrame,
    intervals: pd.DataFrame,
    dist: pd.DataFrame,
    labels: Mapping[str, str],
    first_target: pd.Timestamp,
    last_target: pd.Timestamp,
) -> str:
    n = int(point["n"].iloc[0])
    parts = [
        f"# Backtest summary: horizon = {cfg.horizon} trading day(s)",
        "",
        f"Test targets: {first_target.date()} → {last_target.date()} ({n} trading days), "
        f"{'expanding' if cfg.window is None else f'rolling {cfg.window}-day'} estimation window "
        f"refitted at every origin. DM statistics are against the random walk; "
        "positive = model better; `*`/`**`/`***` = p < 0.10 / 0.05 / 0.01.",
        "",
        "## Point forecasts",
        "",
        point_table(point, labels),
    ]
    if cfg.horizon > 1:
        parts += [
            "",
            f"> **Note on the Christoffersen test at h = {cfg.horizon}.** Consecutive "
            f"{cfg.horizon}-day targets overlap, so interval hits are serially dependent "
            "by construction and the independence null is not valid; only the Kupiec "
            "(unconditional) p-values and the DM tests (which use a HAC variance) "
            "should be read at this horizon.",
        ]
    for level in cfg.levels:
        parts += [
            "",
            f"## {level:.0%} prediction intervals",
            "",
            interval_table(intervals, labels, level),
        ]
    parts += ["", "## Whole-distribution score", "", distribution_table(dist, labels), ""]
    return "\n".join(parts)


def write_frames(frames: Mapping[str, pd.DataFrame], directory: Path) -> Sequence[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in frames.items():
        path = directory / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.5f")
        written.append(path)
    return written
