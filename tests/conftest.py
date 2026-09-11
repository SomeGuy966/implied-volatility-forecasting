from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def simulate_log_vix(n: int, seed: int = 0, phi: float = 0.97, mu: float = 2.9) -> np.ndarray:
    """Persistent AR(1) in logs with GARCH-ish bursts and fat tails; looks like log VIX."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    sigma2 = 0.004
    for t in range(1, n):
        z = rng.standard_t(5) * np.sqrt(3 / 5)  # unit-variance t(5)
        eps = np.sqrt(sigma2) * z
        x[t] = mu * (1 - phi) + phi * x[t - 1] + eps
        sigma2 = 0.0004 + 0.10 * eps**2 + 0.80 * sigma2
    return x


@pytest.fixture(scope="session")
def log_series() -> pd.Series:
    """~5 years of synthetic log VIX on a real NYSE-like trading calendar."""
    idx = pd.bdate_range("2018-01-02", periods=1300)
    # drop a handful of weekdays to mimic exchange holidays
    holidays = idx[[10, 60, 120, 255, 300, 500, 760, 1010, 1250]]
    idx = idx.drop(holidays)
    return pd.Series(simulate_log_vix(len(idx)), index=idx, name="log_close")


@pytest.fixture(scope="session")
def level_series(log_series: pd.Series) -> pd.Series:
    out = np.exp(log_series)
    out.name = "close"
    return out
