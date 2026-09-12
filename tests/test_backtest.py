from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vixcast.backtest import band_cols, quantile_col, walk_forward
from vixcast.config import Config
from vixcast.models import AVAILABLE_MODELS, build_models


def _cfg(**overrides: object) -> Config:
    base: dict[str, object] = {
        "test_start": "2022-06-01",
        "test_end": "2022-07-15",
        "n_jobs": 1,
        "make_plots": False,
    }
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


def test_target_is_h_trading_days_after_origin(log_series: pd.Series) -> None:
    cfg = _cfg(horizon=3, models=("rw",))
    fc = walk_forward(log_series, build_models(["rw"]), cfg)
    pos = {d: i for i, d in enumerate(log_series.index)}
    gaps = [pos[t] - pos[o] for t, o in zip(fc["target_date"], fc["origin_date"], strict=True)]
    assert set(gaps) == {3}
    # positional, not calendar: a weekend/holiday never shrinks the gap
    assert (fc["target_date"] > fc["origin_date"]).all()


def test_random_walk_forecast_equals_level_at_origin(log_series: pd.Series) -> None:
    cfg = _cfg(models=("rw",))
    fc = walk_forward(log_series, build_models(["rw"]), cfg)
    expected = np.exp(log_series.loc[fc["origin_date"]].to_numpy())
    np.testing.assert_allclose(fc["point"].to_numpy(), expected)
    np.testing.assert_allclose(fc["y_true"].to_numpy(), np.exp(log_series.loc[fc["target_date"]]))


def test_output_schema(log_series: pd.Series) -> None:
    cfg = _cfg(models=("rw", "har"))
    fc = walk_forward(log_series, build_models(["rw", "har"]), cfg)
    for p in cfg.quantiles:
        assert quantile_col(p) in fc.columns
    lo, hi = band_cols(0.90)
    assert (fc[lo] <= fc[hi]).all()
    assert (fc[lo] > 0).all(), "level-scale forecasts must be positive"
    assert set(fc["model"]) == {"rw", "har"}
    assert fc.groupby("model").size().nunique() == 1, "every model forecasts the same dates"


@pytest.mark.parametrize("key", AVAILABLE_MODELS)
def test_no_look_ahead(log_series: pd.Series, key: str) -> None:
    """Corrupting every observation after the origin must not change the forecast.

    This is the single most important property of the backtest: it proves each
    forecast is a function of past data only.
    """
    target = log_series.index[1100]
    cfg = _cfg(test_start=str(target.date()), test_end=str(target.date()), models=(key,))
    model = build_models([key])

    clean = walk_forward(log_series, model, cfg)
    assert len(clean) == 1
    origin_pos = log_series.index.get_loc(clean["origin_date"].iloc[0])

    corrupted = log_series.copy()
    rng = np.random.default_rng(123)
    corrupted.iloc[origin_pos + 1 :] += rng.normal(scale=2.0, size=len(corrupted) - origin_pos - 1)
    dirty = walk_forward(corrupted, model, cfg)

    qcols = [c for c in clean.columns if c.startswith("q")]
    np.testing.assert_allclose(
        clean[["point", *qcols]].to_numpy(), dirty[["point", *qcols]].to_numpy()
    )
    assert clean["y_true"].iloc[0] != dirty["y_true"].iloc[0]  # the target itself did change


def test_rolling_window_limits_history(log_series: pd.Series) -> None:
    """With a rolling window the forecast must ignore data older than the window."""
    target = log_series.index[1100]
    cfg = _cfg(
        test_start=str(target.date()), test_end=str(target.date()), models=("har",), window=400
    )
    model = build_models(["har"])
    clean = walk_forward(log_series, model, cfg)
    origin_pos = log_series.index.get_loc(clean["origin_date"].iloc[0])

    corrupted = log_series.copy()
    corrupted.iloc[: origin_pos + 1 - 400] += 5.0
    dirty = walk_forward(corrupted, model, cfg)
    np.testing.assert_allclose(clean["point"].to_numpy(), dirty["point"].to_numpy())


def test_expanding_window_uses_all_history(log_series: pd.Series) -> None:
    target = log_series.index[1100]
    cfg = _cfg(test_start=str(target.date()), test_end=str(target.date()), models=("har",))
    model = build_models(["har"])
    clean = walk_forward(log_series, model, cfg)
    corrupted = log_series.copy()
    corrupted.iloc[:200] += 5.0
    dirty = walk_forward(corrupted, model, cfg)
    assert not np.allclose(clean["point"].to_numpy(), dirty["point"].to_numpy())


def test_empty_test_window_raises(log_series: pd.Series) -> None:
    cfg = _cfg(test_start="2030-01-01", test_end="2030-02-01", models=("rw",))
    with pytest.raises(ValueError):
        walk_forward(log_series, build_models(["rw"]), cfg)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        Config(horizon=0)
    with pytest.raises(ValueError):
        Config(levels=(1.2,))
    with pytest.raises(ValueError):
        Config(window=100, min_train=260)
    assert 0.025 in Config(levels=(0.95,)).quantiles
    assert 0.5 in Config().quantiles
