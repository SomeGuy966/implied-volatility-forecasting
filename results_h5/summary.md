# Backtest summary: horizon = 5 trading day(s)

Test targets: 2024-01-02 → 2026-09-09 (676 trading days), expanding estimation window refitted at every origin. DM statistics are against the random walk; positive = model better; `*`/`**`/`***` = p < 0.10 / 0.05 / 0.01.

## Point forecasts

| Model | RMSE | MAE | RMSE / RW | DM (sq. loss) | DM (abs. loss) |
| :-- | --: | --: | --: | --: | --: |
| Random walk | 3.747 | 2.244 | 1.000 | – | – |
| EWMA | 3.915 | 2.479 | 1.045 | -0.78 | -1.69* |
| HAR | 3.430 | 2.059 | 0.915 | +2.15** | +3.36*** |
| AR(1)-GARCH-t | 3.710 | 2.227 | 0.990 | +0.88 | +0.70 |
| HAR-GARCH-t | 3.452 | 2.035 | 0.921 | +1.58 | +2.59*** |

> **Note on the Christoffersen test at h = 5.** Consecutive 5-day targets overlap, so interval hits are serially dependent by construction and the independence null is not valid; only the Kupiec (unconditional) p-values and the DM tests (which use a HAC variance) should be read at this horizon.

## 80% prediction intervals

| Model | Coverage | Avg width | Interval score | IS / RW | Kupiec p | Christoffersen p | DM (IS) |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Random walk | 79.7% | 6.20 | 11.87 | 1.000 | 0.86 | 0.00 | – |
| EWMA | 82.2% | 7.39 | 12.31 | 1.037 | 0.14 | 0.00 | -0.59 |
| HAR | 78.6% | 5.85 | 10.85 | 0.915 | 0.35 | 0.00 | +2.94*** |
| AR(1)-GARCH-t | 82.4% | 6.86 | 11.40 | 0.960 | 0.11 | 0.00 | +1.25 |
| HAR-GARCH-t | 79.3% | 5.70 | 11.08 | 0.933 | 0.65 | 0.00 | +1.32 |

## 90% prediction intervals

| Model | Coverage | Avg width | Interval score | IS / RW | Kupiec p | Christoffersen p | DM (IS) |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Random walk | 89.3% | 8.71 | 15.83 | 1.000 | 0.58 | 0.00 | – |
| EWMA | 90.8% | 10.18 | 15.90 | 1.004 | 0.47 | 0.00 | -0.06 |
| HAR | 88.9% | 8.17 | 14.46 | 0.913 | 0.35 | 0.00 | +2.46** |
| AR(1)-GARCH-t | 90.4% | 9.46 | 15.18 | 0.959 | 0.74 | 0.00 | +0.89 |
| HAR-GARCH-t | 86.5% | 7.85 | 15.02 | 0.949 | 0.00 | 0.00 | +0.84 |

## 95% prediction intervals

| Model | Coverage | Avg width | Interval score | IS / RW | Kupiec p | Christoffersen p | DM (IS) |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Random walk | 94.1% | 11.64 | 20.85 | 1.000 | 0.29 | 0.00 | – |
| EWMA | 95.3% | 13.75 | 20.77 | 0.996 | 0.75 | 0.00 | +0.05 |
| HAR | 94.2% | 11.05 | 19.35 | 0.928 | 0.37 | 0.00 | +1.93* |
| AR(1)-GARCH-t | 94.4% | 12.25 | 20.03 | 0.961 | 0.47 | 0.00 | +0.75 |
| HAR-GARCH-t | 93.5% | 10.18 | 19.70 | 0.945 | 0.08 | 0.00 | +0.75 |

## Whole-distribution score

| Model | Avg pinball loss | vs RW | DM |
| :-- | --: | --: | --: |
| Random walk | 0.8172 | 1.000 | – |
| EWMA | 0.8785 | 1.075 | -1.27 |
| HAR | 0.7469 | 0.914 | +3.30*** |
| AR(1)-GARCH-t | 0.8052 | 0.985 | +1.12 |
| HAR-GARCH-t | 0.7473 | 0.914 | +2.24** |
