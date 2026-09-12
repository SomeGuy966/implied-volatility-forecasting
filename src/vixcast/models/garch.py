"""GARCH-family forecasters built on the ``arch`` package.

Two specifications share one implementation:

* ``ar_garch``  - AR(1) mean on *log returns* with GARCH(1,1) Student-t errors. This
  treats log VIX as integrated and only models short-run dynamics of the changes.
* ``har_garch`` - HAR mean on the *log level* with GARCH(1,1) Student-t errors. This
  nests the plain HAR (mean reversion) and adds time-varying error variance.

Both are refitted at every forecast origin by maximum likelihood. Data are scaled by
100 before fitting so the optimiser works near unit scale (``arch`` warns otherwise)
and all outputs are mapped back before leaving this module.

Estimation robustness
    GARCH likelihoods have poor local optima, and the optimiser will happily report
    "converged" from one of them (observed in practice: a HAR-GARCH fit with a mean
    forecast of 50 and a 90% band of 1.6-1500). The guard used here is exact rather
    than heuristic: the constant-variance model is nested in GARCH, so the GARCH
    maximum likelihood can never be below the constant-variance fit. Each forecast
    first fits the nested model (fast), uses it for starting values, and rejects any
    GARCH fit whose likelihood falls short of it. Rejected fits retry from ``arch``'s
    own defaults; if that also fails, the nested model's forecast is used and the row
    is flagged ``converged=False`` so the backtest can report it.

Horizons
    For ``horizon == 1`` the forecast distribution is available in closed form as a
    location-scale standardised-t. For longer horizons the AR/HAR recursion makes the
    ``h``-step distribution non-trivial, so it is simulated from the fitted model.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from .base import FloatArray, Forecaster, LogForecast
from .baselines import RandomWalk

Target = Literal["diff", "level"]
_LL_TOL = 1e-6


class ArchGarch(Forecaster):
    expensive = True

    def __init__(
        self,
        key: str,
        label: str,
        mean: Literal["AR", "HARX"],
        lags: int | Sequence[int],
        target: Target,
        p: int = 1,
        q: int = 1,
        dist: Literal["t", "normal", "skewt", "ged"] = "t",
        scale: float = 100.0,
        n_sim: int = 2000,
        seed: int = 0,
        maxiter: int = 1500,
    ) -> None:
        self.key = key
        self.label = label
        self.mean = mean
        self.lags = list(lags) if isinstance(lags, Sequence) else int(lags)
        self.target = target
        self.p, self.q = p, q
        self.dist = dist
        self.scale = scale
        self.n_sim = n_sim
        self.seed = seed
        self.maxiter = maxiter
        self._fallback = RandomWalk()

    def min_history(self, horizon: int) -> int:
        longest_lag = max(self.lags) if isinstance(self.lags, list) else self.lags
        return longest_lag + horizon + 250

    # ------------------------------------------------------------------ fitting
    def _series(self, log_history: FloatArray) -> FloatArray:
        if self.target == "diff":
            return np.diff(log_history) * self.scale
        return log_history * self.scale

    def _mean_model(self, y: FloatArray, volatility: Any) -> Any:
        from arch.univariate import ARX, HARX, GeneralizedError, Normal, SkewStudent, StudentsT

        distribution: Any
        if self.dist == "t":
            distribution = StudentsT()
        elif self.dist == "normal":
            distribution = Normal()
        elif self.dist == "skewt":
            distribution = SkewStudent()
        else:
            distribution = GeneralizedError()
        kwargs = {"lags": self.lags, "volatility": volatility, "distribution": distribution}
        if self.mean == "AR":
            return ARX(y, rescale=False, **kwargs)
        return HARX(y, rescale=False, **kwargs)

    def _fit(self, y: FloatArray) -> tuple[Any, bool]:
        """Return (fitted result, converged)."""
        from arch.univariate import GARCH, ConstantVariance

        nested = self._mean_model(y, ConstantVariance()).fit(disp="off", show_warning=False)
        k = nested.model.num_params  # number of mean-equation parameters
        theta = nested.params.to_numpy()
        mean_start, sigma2, dist_start = theta[:k], theta[k], theta[k + 1 :]
        alphas, betas = [0.1 / self.p] * self.p, [0.8 / self.q] * self.q
        start = np.concatenate(
            [mean_start, [sigma2 * (1.0 - sum(alphas) - sum(betas))], alphas, betas, dist_start]
        )

        garch = self._mean_model(y, GARCH(p=self.p, q=self.q))
        opts = {"maxiter": self.maxiter}
        res = garch.fit(disp="off", show_warning=False, starting_values=start, options=opts)
        if not self._acceptable(res, nested.loglikelihood):
            alt = garch.fit(disp="off", show_warning=False, options=opts)
            if alt.loglikelihood > res.loglikelihood:
                res = alt
        if self._acceptable(res, nested.loglikelihood):
            return res, True
        return nested, False

    @staticmethod
    def _acceptable(res: Any, floor: float) -> bool:
        return (
            int(res.convergence_flag) == 0
            and np.isfinite(res.loglikelihood)
            and res.loglikelihood >= floor - _LL_TOL
        )

    def forecast(
        self, log_history: FloatArray, horizon: int, probs: Sequence[float]
    ) -> LogForecast:
        y = self._series(log_history)
        base = float(log_history[-1]) if self.target == "diff" else 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res, converged = self._fit(y)
                if horizon == 1:
                    point, quantiles = self._analytic_one_step(res, probs)
                else:
                    point, quantiles = self._simulated(res, horizon, probs, len(log_history))
            point = base + point / self.scale
            quantiles = base + quantiles / self.scale
            if not (np.isfinite(point) and np.all(np.isfinite(quantiles))):
                raise FloatingPointError("non-finite forecast")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
            # Estimation failed outright: fall back to the random walk so the backtest
            # keeps a forecast for every date, and flag the row so it can be counted.
            fb = self._fallback.forecast(log_history, horizon, probs)
            return LogForecast(fb.point, fb.probs, fb.quantiles, converged=False)
        return LogForecast(point, tuple(probs), quantiles, converged=converged)

    @staticmethod
    def _analytic_one_step(res: Any, probs: Sequence[float]) -> tuple[float, FloatArray]:
        fc = res.forecast(horizon=1, reindex=False)
        mu = float(fc.mean.iloc[-1, 0])
        sigma = float(np.sqrt(fc.variance.iloc[-1, 0]))
        # ``arch`` distributions are standardised (unit variance), so their ppf is
        # the right multiplier for sigma. Using scipy's raw t.ppf here would inflate
        # every interval by sqrt(nu / (nu - 2)).
        distribution = res.model.distribution
        all_params = res.params.to_numpy()
        dist_params = all_params[len(all_params) - distribution.num_params :]
        z = np.asarray(distribution.ppf(np.asarray(probs), dist_params), dtype=float)
        return mu, mu + sigma * z

    def _simulated(
        self, res: Any, horizon: int, probs: Sequence[float], n_obs: int
    ) -> tuple[float, FloatArray]:
        rng = np.random.RandomState(self.seed + n_obs)  # deterministic per origin
        fc = res.forecast(
            horizon=horizon,
            method="simulation",
            simulations=self.n_sim,
            reindex=False,
            random_state=rng,
        )
        paths = fc.simulations.values[-1]  # (n_sim, horizon) simulated y_{t+1..t+h}
        terminal = paths.sum(axis=1) if self.target == "diff" else paths[:, -1]
        return float(terminal.mean()), np.quantile(terminal, np.asarray(probs))


def ar_garch(seed: int = 0) -> ArchGarch:
    """AR(1)-GARCH(1,1)-t on daily log returns of the VIX."""
    return ArchGarch("ar_garch", "AR(1)-GARCH-t", mean="AR", lags=1, target="diff", seed=seed)


def har_garch(seed: int = 0) -> ArchGarch:
    """HAR(1,5,22)-GARCH(1,1)-t on the log level of the VIX."""
    return ArchGarch(
        "har_garch", "HAR-GARCH-t", mean="HARX", lags=[1, 5, 22], target="level", seed=seed
    )
