from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.offsets import BDay
from arch import arch_model
from scipy.stats import t as student_t

import warnings
from arch.utility.exceptions import ConvergenceWarning

# Quiet down solver / convergence noise from 'arch'
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The optimizer returned code")


# Paths / directories
ROOT = Path(__file__).resolve().parent # sets ROOT to the folder that contains this file
DATA, FIGS, RES = ROOT / "data", ROOT / "figures", ROOT / "results"  # constructs folder pathways


# Train/test windows
TRAIN_START, TRAIN_END = "2015-01-01", "2023-12-31"
TEST_START, TEST_END = "2024-01-01", None

# For two-sided prediction bands: 1 - alpha = {80%, 90%, 95%}
ALPHAS = [0.20, 0.10, 0.05]


# 1) Data loading / preparation
def load_prepare():
    import pandas as pd, numpy as np, yfinance as yf

    # pull data
    df = yf.download("^VIX", start="2010-01-01", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no data for ^VIX.")


    # get 'Close' and force it to a series (handle the 1-col DataFrame case)
    close = df.loc[:, "Close"]
    if isinstance(close, pd.DataFrame):
        # squeeze the single column to a series
        if close.shape[1] != 1:
            raise RuntimeError(f"Unexpected Close shape: {close.shape}")
        close = close.iloc[:, 0]


    s = (close
         .astype(float)  # ensure numeric dtype (needed for log, math, saving)
         .dropna()  # remove any missing closes
         .asfreq("B")  # reindex to Business-day frequency (Mon–Fri only)
         .ffill()  # forward-fill any missing weekdays (e.g., holidays)
         )


    # Series for log-level and log-diff (log-return)
    logv = np.log(s)            # log VIX level (Series)
    dlog = logv.diff().dropna() # log-return (Δ log VIX), more stationary


    # Save artifacts for reproducibility / inspection
    s.to_csv(DATA / "vix.csv")
    prepped = pd.DataFrame({"log_vix": logv, "dlog_vix": dlog})
    prepped.to_csv(DATA / "vix_prepared.csv")

    return s, logv, dlog


# 2) Baseline models (RW, EWMA)
def baselines(level):
    # Build test index
    _, test_idx = split_idx(level.index)

    # Random Walk: forecast = yesterday's level
    # level = original time series that gives raw closing price values
    # .shift(1) shifts all values to the right
    # test_idx = got from split_idx; splits the time series in two depending on the split date
    #
    # The .shift(1) also prevents look-ahead bias by making sure that the forecast for time t
    # only uses values up to and including yesterday and not today
    #
    # For a random walk, the best prediction for the value for today at time t is simply yesterday's
    # value at time t-1
    rw = level.shift(1).loc[test_idx]





    # EWMA baseline on level, shifted to be strictly one-step-ahead
    ewma = level.ewm(alpha=0.05, adjust=False).mean().shift(1).loc[test_idx]


    # Align truth with forecasts and drop any leading NaNs from shifting
    y = level.loc[test_idx]
    df = pd.DataFrame({"y_true": y, "rw": rw, "ewma": ewma}).dropna()

    # Persist
    df.to_csv(RES / "baseline_forecasts.csv")
    return df


# 2a) Train/test split helpers
# Returns (train_idx, test_idx) by slicing the given index using the global date bounds
# TRAIN_START/END and TEST_START/END (open-ended if TEST_END is None).

# train_idx, test_idx are two DateTimeIndex objects
def split_idx(idx):
    # Select train dates within [TRAIN_START, TRAIN_END]
    tr = idx[(idx >= pd.Timestamp(TRAIN_START)) & (idx <= pd.Timestamp(TRAIN_END))]

    # Select test dates from TEST_START to TEST_END (or to end if None)
    te = (
        idx[(idx >= pd.Timestamp(TEST_START))]
        if TEST_END is None
        else idx[(idx >= pd.Timestamp(TEST_START)) & (idx <= pd.Timestamp(TEST_END))]
    )
    return tr, te






# Rolling AR(1)-GARCH(1,1) with Student-t innovations
SCALE = 100.0  # 10 also works; 100 puts the scale near 1 and is often better
def roll_garch(logv, dlog):
    # build test index
    _, test_idx = split_idx(logv.index)

    rows = []

    for date in test_idx:
        # fit on data strictly prior to 'date'
        end = date - BDay(1)

        # pull in-sample log-returns up to yesterday
        y_raw = dlog.loc[:end].dropna()
        if len(y_raw) < 260:  # Require ~1 year of daily obs to fit
            rows.append((date, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        # Rescale returns for better optimizer conditioning
        y = y_raw * SCALE   # <-- rescale

        # AR(1)-GARCH(1,1) with Student-t innovations; keep internal rescale off (we did it)
        am = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", update_freq=0, options={"maxiter": 1500})

        # one-step-ahead forecast (mean and variance) on the rescaled domain
        f = res.forecast(horizon=1, reindex=False)
        mu_sc  = float(f.mean.iloc[-1, 0])                    # predicted Δlog (scaled)
        sig_sc = float(np.sqrt(f.variance.iloc[-1, 0]))       # predicted std dev (scaled)

        # convert forecasts back to original (unscaled) units
        mu  = mu_sc  / SCALE
        sig = sig_sc / SCALE

        # Guardrails: keep numbers finite / sane for intervals
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(sig) or sig <= 0:
            sig = 1e-3
        sig = float(min(sig, 0.5))  # cap sigma to avoid absurdly wide bands

        # Degrees of freedom for Student-t; default to 10 if missing
        try:
            nu = float(getattr(res.distribution, "nu", 10.0))
        except Exception:
            nu = 10.0
        nu = max(nu, 5.0)  # avoid ultra-heavy tails that explode quantiles

        # Build one-step-ahead log-level prediction
        last_log = float(logv.loc[:end].iloc[-1])             # yesterday's log VIX
        log_pred = float(np.clip(last_log + mu, 0.0, 5.5))    # next-day log VIX (bounded)

        # construct prediction bands on log scale, then exponentiate to level
        bands = {}
        for a in ALPHAS:
            # Two-sided t-quantile: 1 - a/2
            q = float(np.clip(student_t.ppf(1 - a/2, df=nu), 1.0, 3.5))

            lo_log = float(np.clip(log_pred - q * sig, 0.0, 5.5))
            hi_log = float(np.clip(log_pred + q * sig, 0.0, 5.5))

            # Store on level scale and key by nominal % (80/90/95)
            bands[int((1 - a) * 100)] = (np.exp(lo_log), np.exp(hi_log))

        # Point forecast on level scale
        level_pred = float(np.exp(log_pred))

        # Accumulate row for this forecast date
        rows.append((
            date, level_pred,
            bands[80][0], bands[80][1],
            bands[90][0], bands[90][1],
            bands[95][0], bands[95][1]
        ))

    # Assemble rolling forecast frame
    cols = ["date", "garch_pred", "lo80", "hi80", "lo90", "hi90", "lo95", "hi95"]
    out = pd.DataFrame(rows, columns=cols).set_index("date")

    # Join realized level for evaluation (exp of log level)
    y_true = np.exp(logv.loc[out.index])
    out = out.join(y_true.rename("y_true")).dropna()

    # Persist
    out.to_csv(RES / "garch_forecasts.csv")
    return out



# point forecasts
def evaluate_mean(df_all):
    # Realized values
    y = df_all["y_true"].values

    rows = []
    for c in ["rw", "ewma", "garch_pred"]:
        pred = df_all[c].values

        # Standard regression metrics on level scale
        mse  = mean_squared_error(y, pred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y, pred))

        rows.append({"model": c, "RMSE": rmse, "MAE": mae})

    # sort by RMSE (best first) and persist
    met = pd.DataFrame(rows).sort_values("RMSE")
    met.to_csv(RES / "metrics_mean.csv", index=False)
    return met

# prediction intervals
def evaluate_intervals(gdf):
    cov, wdth = [], []

    for p in (80, 90, 95):
        lo, hi = f"lo{p}", f"hi{p}"

        # Coverage: fraction of y_true within [lo, hi]
        inside = ((gdf["y_true"] >= gdf[lo]) & (gdf["y_true"] <= gdf[hi])).mean()

        # Average band width
        width = (gdf[hi] - gdf[lo]).mean()

        cov.append({"band": p, "coverage": inside})
        wdth.append({"band": p, "avg_width": width})

    # Persist coverage and width tables
    cov = pd.DataFrame(cov)
    wdth = pd.DataFrame(wdth)

    cov.to_csv(RES / "metrics_coverage.csv", index=False)
    wdth.to_csv(RES / "metrics_width.csv", index=False)

    return cov, wdth



# Plotting
def plot_overlay(df_all):
    # overlay actual and point forecasts across the test window
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df_all.index, df_all["y_true"],      label="Actual")
    ax.plot(df_all.index, df_all["garch_pred"],  label="AR(1)-GARCH mean")
    ax.plot(df_all.index, df_all["rw"],          label="RW")
    ax.plot(df_all.index, df_all["ewma"],        label="EWMA")

    ax.legend()
    ax.set_title("VIX: Actual vs Forecasts (Test)")

    fig.tight_layout()
    fig.savefig(FIGS / "test_overlay_all.png", dpi=200)
    # (no plt.close here; we show later at the bottom)
def plot_band(gdf, band=90):
    # Plot realized series with shaded prediction band at the chosen level
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(gdf.index, gdf["y_true"],     label="Actual")
    ax.plot(gdf.index, gdf["garch_pred"], label="Mean")

    ax.fill_between(
        gdf.index,
        gdf[f"lo{band}"],
        gdf[f"hi{band}"],
        alpha=0.25,
        label=f"{band}% PI"
    )

    ax.legend()
    ax.set_title(f"VIX {band}% Prediction Band (AR-GARCH)")

    fig.tight_layout()
    fig.savefig(FIGS / f"band_{band}.png", dpi=200)
    plt.close(fig)  # close individual band figures to keep memory tidy






def main():
    # load and prep data
    level, logv, dlog = load_prepare()

    # baselines on the level series
    base  = baselines(level)

    # rolling AR-GARCH on log-returns, bands on level
    garch = roll_garch(logv, dlog)

    # join everything for evaluation on common test dates
    joined = base.join(
        garch[["garch_pred", "lo80", "hi80", "lo90", "hi90", "lo95", "hi95"]],
        how="inner"
    )

    # Point forecast metrics
    met_mean = evaluate_mean(joined)

    # Interval metrics (coverage, average width)
    cov, wdth = evaluate_intervals(joined)

    # Figures
    plot_overlay(joined)
    for b in (80, 90, 95):
        plot_band(garch, band=b)

    # Console summary
    print("\nMean metrics:\n", met_mean.to_string(index=False))
    print("\nCoverage:\n", cov.to_string(index=False))
    print("\nAvg width:\n", wdth.to_string(index=False))
    print(f"\nArtifacts → {RES}  {FIGS}")

if __name__ == "__main__":
    main()

    # Keep matplotlib windows open (useful when running outside notebooks/CI)
    import matplotlib
    import matplotlib.pyplot as plt

    print("Matplotlib backend:", matplotlib.get_backend())
    plt.show(block=True)  # keep the windows open until they close
