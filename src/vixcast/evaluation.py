"""Point-forecast and interval-forecast evaluation.

Point forecasts
    RMSE / MAE on the VIX level, plus a Diebold-Mariano (1995) test of equal
    predictive accuracy against the random walk, using a HAC variance for
    multi-step horizons and the Harvey-Leybourne-Newbold (1997) small-sample
    correction.

Interval forecasts
    For each nominal level: empirical coverage, average width, the Winkler /
    interval score (Gneiting & Raftery, 2007), the Kupiec (1995) unconditional
    coverage test and the Christoffersen (1998) conditional coverage test. The
    interval score is a proper scoring rule, so a DM test on it gives a principled
    "are these intervals better than the benchmark's?" answer.

Whole distribution
    Pinball loss averaged over a dense quantile grid (a discretised CRPS) and a
    calibration curve of nominal vs empirical central-interval coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from .backtest import band_cols, quantile_col
from .models.base import FloatArray

# ----------------------------------------------------------------------------- losses


def squared_error(y: FloatArray, f: FloatArray) -> FloatArray:
    return (y - f) ** 2


def absolute_error(y: FloatArray, f: FloatArray) -> FloatArray:
    return np.abs(y - f)


def interval_score(y: FloatArray, lo: FloatArray, hi: FloatArray, level: float) -> FloatArray:
    """Winkler / interval score for a central ``level`` prediction interval.

    Width plus a penalty of ``2/alpha`` times the distance by which the realisation
    misses the interval. Lower is better; it is minimised in expectation by the true
    quantiles, i.e. it is a proper scoring rule.
    """
    alpha = 1.0 - level
    width = hi - lo
    below = np.where(y < lo, lo - y, 0.0)
    above = np.where(y > hi, y - hi, 0.0)
    return width + (2.0 / alpha) * (below + above)


def pinball_loss(y: FloatArray, q: FloatArray, tau: float) -> FloatArray:
    diff = y - q
    return np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)


# ------------------------------------------------------------------ Diebold-Mariano


@dataclass(frozen=True)
class DMResult:
    statistic: float
    pvalue: float
    n: int


def diebold_mariano(
    loss_benchmark: FloatArray, loss_model: FloatArray, horizon: int = 1
) -> DMResult:
    """Two-sided DM test of H0: E[loss_benchmark - loss_model] = 0.

    A *positive* statistic means the model beats the benchmark. The long-run variance
    of the loss differential uses a Bartlett kernel with ``horizon - 1`` lags (the
    differential is MA(h-1) under optimal h-step forecasts) and the statistic carries
    the HLN correction with a t(n-1) reference distribution.
    """
    d = np.asarray(loss_benchmark, dtype=float) - np.asarray(loss_model, dtype=float)
    n = d.size
    if n < 10:
        raise ValueError("need at least 10 observations for a DM test")
    d_bar = d.mean()
    d_c = d - d_bar
    lrv = float(d_c @ d_c) / n
    for k in range(1, horizon):
        gamma_k = float(d_c[k:] @ d_c[:-k]) / n
        lrv += 2.0 * (1.0 - k / horizon) * gamma_k
    if lrv <= 0.0:
        return DMResult(float("nan"), float("nan"), n)
    dm = d_bar / np.sqrt(lrv / n)
    hln = dm * np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    p = 2.0 * stats.t.sf(abs(hln), df=n - 1)
    return DMResult(float(hln), float(p), n)


# ---------------------------------------------------------------- coverage tests


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


@dataclass(frozen=True)
class CoverageTests:
    lr_uc: float
    p_uc: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float


def coverage_tests(hits: NDArray[np.bool_], level: float) -> CoverageTests:
    """Kupiec unconditional and Christoffersen conditional coverage tests.

    ``hits`` is the indicator that the realisation fell inside the interval, so its
    expectation under correct calibration is ``level``. The likelihood ratios are
    invariant to relabelling hits as violations.
    """
    h = np.asarray(hits, dtype=bool)
    n = h.size
    n1 = int(h.sum())
    n0 = n - n1
    pi_hat = n1 / n

    ll_null = _xlogy(n0, 1.0 - level) + _xlogy(n1, level)
    ll_alt = _xlogy(n0, 1.0 - pi_hat) + _xlogy(n1, pi_hat)
    lr_uc = -2.0 * (ll_null - ll_alt)
    p_uc = float(stats.chi2.sf(lr_uc, df=1))

    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n - 1)
    ll_ind_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll_ind_alt = (
        _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    )
    lr_ind = -2.0 * (ll_ind_null - ll_ind_alt)
    p_ind = float(stats.chi2.sf(lr_ind, df=1))

    lr_cc = lr_uc + lr_ind
    p_cc = float(stats.chi2.sf(lr_cc, df=2))
    return CoverageTests(lr_uc, p_uc, lr_ind, p_ind, lr_cc, p_cc)


# ------------------------------------------------------------------ table builders


def _aligned(fc: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Restrict to target dates every requested model forecast, so tests are paired."""
    sub = fc[fc["model"].isin(models)]
    counts = sub.groupby("target_date")["model"].nunique()
    common = counts[counts == len(models)].index
    return sub[sub["target_date"].isin(common)].sort_values(["model", "target_date"])


def point_metrics(fc: pd.DataFrame, models: Sequence[str], benchmark: str = "rw") -> pd.DataFrame:
    """One row per model: error statistics and DM tests against ``benchmark``."""
    if benchmark not in models:
        raise ValueError("benchmark must be one of the evaluated models")
    df = _aligned(fc, models)
    h = int(df["horizon"].iloc[0])
    bench = df[df["model"] == benchmark]
    y = bench["y_true"].to_numpy()
    se_b = squared_error(y, bench["point"].to_numpy())
    ae_b = absolute_error(y, bench["point"].to_numpy())
    rmse_b = float(np.sqrt(se_b.mean()))

    rows = []
    for key in models:
        m = df[df["model"] == key]
        f = m["point"].to_numpy()
        se, ae = squared_error(y, f), absolute_error(y, f)
        row: dict[str, object] = {
            "model": key,
            "n": int(y.size),
            "rmse": float(np.sqrt(se.mean())),
            "mae": float(ae.mean()),
            "rmse_ratio": float(np.sqrt(se.mean()) / rmse_b),
            "non_converged": int((~m["converged"].to_numpy(dtype=bool)).sum()),
        }
        if key != benchmark:
            dm_se = diebold_mariano(se_b, se, h)
            dm_ae = diebold_mariano(ae_b, ae, h)
            row.update(
                dm_stat_se=dm_se.statistic,
                dm_p_se=dm_se.pvalue,
                dm_stat_ae=dm_ae.statistic,
                dm_p_ae=dm_ae.pvalue,
            )
        else:
            row.update(dm_stat_se=np.nan, dm_p_se=np.nan, dm_stat_ae=np.nan, dm_p_ae=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def interval_metrics(
    fc: pd.DataFrame,
    models: Sequence[str],
    levels: Sequence[float],
    benchmark: str = "rw",
) -> pd.DataFrame:
    """One row per (model, level): calibration, sharpness, proper score and tests."""
    df = _aligned(fc, models)
    h = int(df["horizon"].iloc[0])
    bench = df[df["model"] == benchmark]
    y = bench["y_true"].to_numpy()

    rows = []
    for level in levels:
        lo_col, hi_col = band_cols(level)
        is_b = interval_score(y, bench[lo_col].to_numpy(), bench[hi_col].to_numpy(), level)
        for key in models:
            m = df[df["model"] == key]
            lo, hi = m[lo_col].to_numpy(), m[hi_col].to_numpy()
            hits = (y >= lo) & (y <= hi)
            is_m = interval_score(y, lo, hi, level)
            tests = coverage_tests(hits, level)
            dm = diebold_mariano(is_b, is_m, h) if key != benchmark else None
            rows.append(
                {
                    "model": key,
                    "level": level,
                    "n": int(y.size),
                    "coverage": float(hits.mean()),
                    "avg_width": float((hi - lo).mean()),
                    "interval_score": float(is_m.mean()),
                    "is_ratio": float(is_m.mean() / is_b.mean()),
                    "kupiec_p": tests.p_uc,
                    "independence_p": tests.p_ind,
                    "christoffersen_p": tests.p_cc,
                    "dm_stat_is": dm.statistic if dm else np.nan,
                    "dm_p_is": dm.pvalue if dm else np.nan,
                }
            )
    return pd.DataFrame(rows)


def distribution_metrics(
    fc: pd.DataFrame, models: Sequence[str], probs: Sequence[float], benchmark: str = "rw"
) -> pd.DataFrame:
    """Average pinball loss over the quantile grid (a discretised CRPS) per model."""
    df = _aligned(fc, models)
    h = int(df["horizon"].iloc[0])
    grid = [p for p in probs if 0.0 < p < 1.0]

    def _mean_pinball(m: pd.DataFrame) -> FloatArray:
        y = m["y_true"].to_numpy()
        losses = [pinball_loss(y, m[quantile_col(p)].to_numpy(), p) for p in grid]
        return np.mean(np.vstack(losses), axis=0)

    pb_b = _mean_pinball(df[df["model"] == benchmark])
    rows = []
    for key in models:
        pb = _mean_pinball(df[df["model"] == key])
        dm = diebold_mariano(pb_b, pb, h) if key != benchmark else None
        rows.append(
            {
                "model": key,
                "n": int(pb.size),
                "pinball": float(pb.mean()),
                "pinball_ratio": float(pb.mean() / pb_b.mean()),
                "dm_stat": dm.statistic if dm else np.nan,
                "dm_p": dm.pvalue if dm else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_curve(
    fc: pd.DataFrame, models: Sequence[str], probs: Sequence[float]
) -> pd.DataFrame:
    """Nominal vs empirical coverage of central intervals built from symmetric quantile pairs."""
    df = _aligned(fc, models)
    lower = sorted(p for p in probs if p < 0.5 and round(1.0 - p, 4) in set(probs))
    rows = []
    for key in models:
        m = df[df["model"] == key]
        y = m["y_true"].to_numpy()
        for p in lower:
            lo = m[quantile_col(p)].to_numpy()
            hi = m[quantile_col(round(1.0 - p, 4))].to_numpy()
            rows.append(
                {
                    "model": key,
                    "nominal": round(1.0 - 2 * p, 4),
                    "empirical": float(((y >= lo) & (y <= hi)).mean()),
                }
            )
    return pd.DataFrame(rows)


def rolling_rmse_ratio(
    fc: pd.DataFrame, models: Sequence[str], window: int = 63, benchmark: str = "rw"
) -> pd.DataFrame:
    """Rolling RMSE of each model divided by the benchmark's, indexed by target date."""
    df = _aligned(fc, [benchmark, *(m for m in models if m != benchmark)])
    bench = df[df["model"] == benchmark].set_index("target_date")
    se_b = squared_error(bench["y_true"].to_numpy(), bench["point"].to_numpy())
    out = pd.DataFrame(index=bench.index)
    denom = pd.Series(se_b, index=bench.index).rolling(window).mean()
    for key in models:
        if key == benchmark:
            continue
        m = df[df["model"] == key].set_index("target_date")
        se = squared_error(m["y_true"].to_numpy(), m["point"].to_numpy())
        out[key] = np.sqrt(pd.Series(se, index=m.index).rolling(window).mean() / denom)
    return out.dropna()


def cumulative_loss_difference(
    fc: pd.DataFrame, models: Sequence[str], benchmark: str = "rw"
) -> pd.DataFrame:
    """Cumulative (benchmark SE - model SE); rising means the model is winning."""
    df = _aligned(fc, [benchmark, *(m for m in models if m != benchmark)])
    bench = df[df["model"] == benchmark].set_index("target_date")
    se_b = squared_error(bench["y_true"].to_numpy(), bench["point"].to_numpy())
    out = pd.DataFrame(index=bench.index)
    for key in models:
        if key == benchmark:
            continue
        m = df[df["model"] == key].set_index("target_date")
        se = squared_error(m["y_true"].to_numpy(), m["point"].to_numpy())
        out[key] = np.cumsum(se_b - se)
    return out
