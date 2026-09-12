"""Load, validate and cache the VIX close series.

Design note: the series is kept on its *actual trading-day* index. An earlier
version of this project reindexed to a Monday-Friday calendar and forward-filled
holidays, which silently inserted ~20 test days per year on which the random-walk
forecast is scored as exactly right. Never do that.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def fetch_close(ticker: str, start: str) -> pd.Series:
    """Download daily closes from Yahoo Finance as a float Series on a DatetimeIndex."""
    import yfinance as yf  # imported lazily so the package works offline

    raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # multi-index columns in newer yfinance
        close = close.iloc[:, 0]
    series = close.astype(float).dropna()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    series.name = "close"
    return validate_series(series)


def validate_series(series: pd.Series) -> pd.Series:
    """Enforce the invariants every downstream component relies on."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by DatetimeIndex")
    if series.index.has_duplicates:
        raise ValueError("duplicate dates in series")
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    if series.isna().any():
        raise ValueError("NaNs in series")
    if (series <= 0).any():
        raise ValueError("non-positive values in series; log transform would fail")
    return series


def read_cache(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = df.iloc[:, 0].astype(float)
    series.index = pd.DatetimeIndex(series.index)
    series.name = "close"
    return validate_series(series)


def write_cache(series: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_frame().rename_axis("date").to_csv(path, float_format="%.6f")


def load_close(
    ticker: str,
    start: str,
    cache_path: Path,
    refresh: bool = False,
) -> pd.Series:
    """Return the close series, preferring the on-disk cache for reproducibility.

    With ``refresh=True`` (or no cache present) the data is downloaded and the cache
    rewritten. The cache is committed to the repo so that CI and reviewers can run the
    full pipeline without network access.
    """
    if cache_path.exists() and not refresh:
        series = read_cache(cache_path)
        log.info("Loaded %d closes from cache %s", len(series), cache_path)
        return series
    series = fetch_close(ticker, start)
    write_cache(series, cache_path)
    log.info("Downloaded %d closes for %s and wrote %s", len(series), ticker, cache_path)
    # Read back through the cache so a fresh download and a cached run see
    # bit-identical inputs (the CSV rounds to 6 dp).
    return read_cache(cache_path)


def to_log(level: pd.Series) -> pd.Series:
    """Log level; all models operate in log space so level forecasts stay positive."""
    out: pd.Series = np.log(level)
    out.name = "log_close"
    return out
