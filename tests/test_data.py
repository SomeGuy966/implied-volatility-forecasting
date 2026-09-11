from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vixcast.data import load_close, read_cache, to_log, validate_series, write_cache


def test_cache_round_trip(tmp_path: Path, level_series: pd.Series) -> None:
    path = tmp_path / "vix.csv"
    write_cache(level_series, path)
    back = read_cache(path)
    pd.testing.assert_index_equal(back.index, level_series.index, check_names=False)
    np.testing.assert_allclose(back.to_numpy(), level_series.to_numpy(), rtol=0, atol=1e-6)


def test_load_prefers_cache_and_never_touches_network(
    tmp_path: Path, level_series: pd.Series
) -> None:
    path = tmp_path / "vix.csv"
    write_cache(level_series, path)
    out = load_close("^NOTATICKER", "2000-01-01", path, refresh=False)
    assert len(out) == len(level_series)


def test_holidays_are_not_forward_filled(level_series: pd.Series) -> None:
    """The index must be the actual trading calendar: no synthetic weekday rows."""
    full_bdays = pd.bdate_range(level_series.index[0], level_series.index[-1])
    assert len(level_series) < len(full_bdays)
    # and no run of identical consecutive values was manufactured by the loader
    repeats = (level_series == level_series.shift(1)).sum()
    assert repeats <= 2


@pytest.mark.parametrize(
    "bad",
    [
        pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-01"])),
        pd.Series([1.0, np.nan], index=pd.to_datetime(["2020-01-01", "2020-01-02"])),
        pd.Series([1.0, -2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"])),
        pd.Series([1.0, 2.0], index=[0, 1]),
    ],
)
def test_validate_rejects_bad_series(bad: pd.Series) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_series(bad)


def test_validate_sorts_unordered_index() -> None:
    s = pd.Series([2.0, 1.0], index=pd.to_datetime(["2020-01-02", "2020-01-01"]))
    out = validate_series(s)
    assert out.index.is_monotonic_increasing
    assert out.iloc[0] == 1.0


def test_to_log_inverts_exp(level_series: pd.Series, log_series: pd.Series) -> None:
    np.testing.assert_allclose(to_log(level_series).to_numpy(), log_series.to_numpy())
