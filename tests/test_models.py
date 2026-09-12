from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vixcast.models import AVAILABLE_MODELS, EWMA, HAR, Forecaster, RandomWalk, build_models
from vixcast.models.base import LogForecast, empirical_quantiles, h_step_targets
from vixcast.models.har import har_design

PROBS = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)


@pytest.fixture(params=AVAILABLE_MODELS)
def model(request: pytest.FixtureRequest) -> Forecaster:
    return build_models([request.param])[0]


@pytest.mark.parametrize("horizon", [1, 5])
def test_every_model_returns_a_sane_forecast(
    model: Forecaster, log_series: pd.Series, horizon: int
) -> None:
    hist = log_series.to_numpy()[:1000]
    fc = model.forecast(hist, horizon, PROBS)
    assert isinstance(fc, LogForecast)
    assert np.isfinite(fc.point)
    assert np.all(np.diff(fc.quantiles) >= 0), "quantiles must be non-decreasing in p"
    # a one-day change in log VIX of more than 1.0 (x2.7) is not a plausible centre
    assert abs(fc.point - hist[-1]) < 1.0
    # the median forecast should sit inside the 95% band
    assert fc.quantiles[0] <= fc.point <= fc.quantiles[-1]


def test_models_are_picklable(model: Forecaster) -> None:
    import pickle

    assert pickle.loads(pickle.dumps(model)).key == model.key


def test_random_walk_point_is_last_value(log_series: pd.Series) -> None:
    hist = log_series.to_numpy()[:500]
    fc = RandomWalk().forecast(hist, 1, PROBS)
    assert fc.point == hist[-1]


def test_random_walk_intervals_widen_with_horizon(log_series: pd.Series) -> None:
    hist = log_series.to_numpy()[:800]
    w1 = np.ptp(RandomWalk().forecast(hist, 1, PROBS).quantiles)
    w5 = np.ptp(RandomWalk().forecast(hist, 5, PROBS).quantiles)
    assert w5 > w1


def test_ewma_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        EWMA(alpha=0.0)


def test_har_design_matches_naive_rolling_means() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    X = har_design(x, (1, 5, 22))
    assert X.shape == (200 - 22 + 1, 4)
    # last row = features at the final observation
    assert X[-1, 0] == 1.0
    assert X[-1, 1] == pytest.approx(x[-1])
    assert X[-1, 2] == pytest.approx(x[-5:].mean())
    assert X[-1, 3] == pytest.approx(x[-22:].mean())
    # first row = features at t = 21
    assert X[0, 3] == pytest.approx(x[:22].mean())


def test_har_recovers_ar1_coefficient() -> None:
    """On a pure AR(1) the HAR should put nearly all weight on the 1-day lag."""
    rng = np.random.default_rng(2)
    n, phi = 5000, 0.9
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(scale=0.1)
    X = har_design(x, (1, 5, 22))
    beta, *_ = np.linalg.lstsq(X[:-1], x[22:], rcond=None)
    assert beta[1] == pytest.approx(phi, abs=0.05)
    assert abs(beta[2]) < 0.1 and abs(beta[3]) < 0.1


def test_har_rejects_duplicate_lags() -> None:
    with pytest.raises(ValueError):
        HAR(lags=(1, 5, 5))


def test_h_step_targets_alignment() -> None:
    x = np.arange(10.0)
    a, b = h_step_targets(x, 3)
    assert np.all(b - a == 3)
    assert len(a) == 7


def test_empirical_quantiles_needs_a_sample() -> None:
    with pytest.raises(ValueError):
        empirical_quantiles(np.zeros(3), PROBS)


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(KeyError):
        build_models(["nope"])


def test_garch_flags_non_convergence_instead_of_raising() -> None:
    """A pathological constant series cannot be fitted; the model must degrade gracefully."""
    from vixcast.models import ar_garch

    hist = np.full(600, 3.0)
    hist[::7] += 1e-6  # not exactly constant so the empirical fallback has a sample
    fc = ar_garch().forecast(hist, 1, PROBS)
    assert np.isfinite(fc.point)
    assert fc.converged is False
