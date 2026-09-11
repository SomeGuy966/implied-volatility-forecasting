from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vixcast import evaluation as ev
from vixcast.backtest import walk_forward
from vixcast.config import Config
from vixcast.models import build_models

# --------------------------------------------------------------------- DM test


def test_dm_is_zero_for_identical_losses() -> None:
    loss = np.random.default_rng(0).exponential(size=200)
    res = ev.diebold_mariano(loss, loss.copy())
    assert np.isnan(res.statistic)  # zero long-run variance => undefined, not a false positive


def test_dm_sign_and_significance() -> None:
    rng = np.random.default_rng(0)
    bench = rng.exponential(size=500) + 0.5
    better = bench * 0.7 + rng.normal(scale=0.05, size=500)
    res = ev.diebold_mariano(bench, better)
    assert res.statistic > 0 and res.pvalue < 0.01
    flipped = ev.diebold_mariano(better, bench)
    assert flipped.statistic == pytest.approx(-res.statistic)


def test_dm_has_correct_size_under_null() -> None:
    """Rejection rate at 5% should be about 5% when the two loss series are exchangeable."""
    rng = np.random.default_rng(42)
    rejections = 0
    trials = 400
    for _ in range(trials):
        a = rng.exponential(size=150)
        b = rng.exponential(size=150)
        if ev.diebold_mariano(a, b).pvalue < 0.05:
            rejections += 1
    rate = rejections / trials
    assert 0.02 < rate < 0.09


def test_dm_hln_reduces_to_plain_dm_when_h_is_one() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.exponential(size=100), rng.exponential(size=100)
    d = a - b
    plain = d.mean() / np.sqrt(d.var() / len(d))
    hln = ev.diebold_mariano(a, b, horizon=1).statistic
    assert hln == pytest.approx(plain * np.sqrt((len(d) + 1 - 2) / len(d)))


# ------------------------------------------------------------- coverage tests


def test_kupiec_accepts_exact_coverage_and_rejects_bad_coverage() -> None:
    n = 1000
    hits = np.zeros(n, dtype=bool)
    hits[:900] = True
    np.random.default_rng(0).shuffle(hits)
    ok = ev.coverage_tests(hits, 0.90)
    assert ok.p_uc > 0.9
    hits_bad = np.zeros(n, dtype=bool)
    hits_bad[:800] = True
    np.random.default_rng(0).shuffle(hits_bad)
    assert ev.coverage_tests(hits_bad, 0.90).p_uc < 1e-6


def test_kupiec_matches_closed_form() -> None:
    hits = np.array([True] * 88 + [False] * 12)
    res = ev.coverage_tests(hits, 0.90)
    n1, n0, p, pi = 88, 12, 0.90, 0.88
    lr = -2 * ((n0 * np.log(1 - p) + n1 * np.log(p)) - (n0 * np.log(1 - pi) + n1 * np.log(pi)))
    assert res.lr_uc == pytest.approx(lr)
    assert res.p_uc == pytest.approx(stats.chi2.sf(lr, 1))


def test_christoffersen_detects_clustered_misses() -> None:
    # 90% coverage overall, but misses come in one block => independence rejected
    hits = np.ones(500, dtype=bool)
    hits[100:150] = False
    res = ev.coverage_tests(hits, 0.90)
    assert res.p_uc > 0.5  # coverage is fine ...
    assert res.p_ind < 1e-6  # ... but the misses are not independent
    assert res.p_cc < 1e-6


def test_coverage_tests_handle_degenerate_all_hits() -> None:
    res = ev.coverage_tests(np.ones(50, dtype=bool), 0.90)
    assert np.isfinite(res.lr_uc) and 0 <= res.p_uc <= 1


# ---------------------------------------------------------------- scoring rules


def test_interval_score_is_width_when_covered_and_penalises_misses() -> None:
    y = np.array([10.0, 10.0, 10.0])
    lo = np.array([9.0, 11.0, 9.0])
    hi = np.array([11.0, 12.0, 9.5])
    s = ev.interval_score(y, lo, hi, 0.90)
    assert s[0] == pytest.approx(2.0)
    assert s[1] == pytest.approx(1.0 + 20.0 * 1.0)
    assert s[2] == pytest.approx(0.5 + 20.0 * 0.5)


def test_interval_score_is_proper() -> None:
    """The true quantiles minimise the expected interval score."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=200_000)
    true_lo, true_hi = stats.norm.ppf(0.05), stats.norm.ppf(0.95)
    best = ev.interval_score(y, np.full_like(y, true_lo), np.full_like(y, true_hi), 0.90).mean()
    for shift in (-0.3, 0.3):
        worse = ev.interval_score(
            y, np.full_like(y, true_lo + shift), np.full_like(y, true_hi + shift), 0.90
        ).mean()
        assert worse > best
    narrower = ev.interval_score(
        y, np.full_like(y, true_lo * 0.7), np.full_like(y, true_hi * 0.7), 0.90
    ).mean()
    assert narrower > best


def test_pinball_loss_is_minimised_at_the_true_quantile() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=100_000)
    tau = 0.9
    q_true = stats.norm.ppf(tau)
    best = ev.pinball_loss(y, np.full_like(y, q_true), tau).mean()
    assert ev.pinball_loss(y, np.full_like(y, q_true - 0.3), tau).mean() > best
    assert ev.pinball_loss(y, np.full_like(y, q_true + 0.3), tau).mean() > best


# ------------------------------------------------------------ table builders


@pytest.fixture(scope="module")
def forecasts(log_series: pd.Series) -> tuple[pd.DataFrame, Config]:
    cfg = Config(
        test_start="2022-01-03", test_end="2022-12-30", models=("rw", "ewma", "har"), n_jobs=1
    )
    return walk_forward(log_series, build_models(list(cfg.models)), cfg), cfg


def test_point_metrics_table(forecasts: tuple[pd.DataFrame, Config]) -> None:
    fc, cfg = forecasts
    tbl = ev.point_metrics(fc, list(cfg.models), "rw").set_index("model")
    assert tbl.loc["rw", "rmse_ratio"] == pytest.approx(1.0)
    assert np.isnan(tbl.loc["rw", "dm_stat_se"])
    assert tbl.loc["ewma", "rmse"] > tbl.loc["rw", "rmse"]  # a smoother lags a persistent series
    assert tbl.loc["ewma", "dm_stat_se"] < 0


def test_interval_metrics_table(forecasts: tuple[pd.DataFrame, Config]) -> None:
    fc, cfg = forecasts
    tbl = ev.interval_metrics(fc, list(cfg.models), cfg.levels, "rw")
    assert set(tbl["level"]) == set(cfg.levels)
    assert ((tbl["coverage"] >= 0) & (tbl["coverage"] <= 1)).all()
    assert (tbl["avg_width"] > 0).all()
    wider = (
        tbl.sort_values(["model", "level"])
        .groupby("model")["avg_width"]
        .apply(lambda s: s.is_monotonic_increasing)
    )
    assert wider.all(), "higher nominal level must give wider bands"


def test_calibration_curve_shape(forecasts: tuple[pd.DataFrame, Config]) -> None:
    fc, cfg = forecasts
    curve = ev.calibration_curve(fc, list(cfg.models), cfg.quantiles)
    for _, sub in curve.groupby("model"):
        sub = sub.sort_values("nominal")
        assert sub["empirical"].is_monotonic_increasing
        assert sub["nominal"].iloc[-1] == pytest.approx(max(cfg.levels))
        assert sub["nominal"].iloc[0] == pytest.approx(0.10)


def test_paired_evaluation_drops_unmatched_dates(forecasts: tuple[pd.DataFrame, Config]) -> None:
    fc, cfg = forecasts
    trimmed = fc[~((fc["model"] == "har") & (fc["target_date"] == fc["target_date"].max()))]
    tbl = ev.point_metrics(trimmed, list(cfg.models), "rw")
    assert tbl["n"].nunique() == 1
    assert tbl["n"].iloc[0] == fc["target_date"].nunique() - 1


def test_relative_performance_frames_include_benchmark_even_if_omitted(
    forecasts: tuple[pd.DataFrame, Config],
) -> None:
    fc, _ = forecasts
    ratio = ev.rolling_rmse_ratio(fc, ["har", "ewma"], window=20, benchmark="rw")
    assert list(ratio.columns) == ["har", "ewma"]
    assert len(ratio) > 0 and np.isfinite(ratio.to_numpy()).all()
    cum = ev.cumulative_loss_difference(fc, ["har"], benchmark="rw")
    assert list(cum.columns) == ["har"]
    assert len(cum) == fc["target_date"].nunique()
