# Backtest summary: horizon = 1 trading day(s)

Test targets: 2024-01-02 → 2026-09-09 (676 trading days), expanding estimation window refitted at every origin. DM statistics are against the random walk; positive = model better; `*`/`**`/`***` = p < 0.10 / 0.05 / 0.01.

## Point forecasts

| Model | RMSE | MAE | RMSE / RW | DM (sq. loss) | DM (abs. loss) |
| :-- | --: | --: | --: | --: | --: |
| Random walk | 1.949 | 1.096 | 1.000 | – | – |
| EWMA | 3.502 | 2.161 | 1.797 | -4.34*** | -12.66*** |
| HAR | 1.900 | 1.073 | 0.975 | +0.98 | +2.09** |
| AR(1)-GARCH-t | 1.935 | 1.080 | 0.993 | +0.75 | +2.68*** |
| HAR-GARCH-t | 1.899 | 1.056 | 0.975 | +0.69 | +2.40** |

## 80% prediction intervals

| Model | Coverage | Avg width | Interval score | IS / RW | Kupiec p | Christoffersen p | DM (IS) |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Random walk | 79.3% | 2.95 | 5.92 | 1.000 | 0.65 | 0.01 | – |
| EWMA | 81.8% | 6.46 | 10.73 | 1.813 | 0.24 | 0.00 | -7.74*** |
| HAR | 78.7% | 2.90 | 5.80 | 0.980 | 0.40 | 0.00 | +1.40 |
| AR(1)-GARCH-t | 79.6% | 3.19 | 5.53 | 0.935 | 0.79 | 0.86 | +2.50** |
| HAR-GARCH-t | 80.2% | 3.08 | 5.47 | 0.925 | 0.91 | 0.67 | +2.25** |

## 90% prediction intervals

| Model | Coverage | Avg width | Interval score | IS / RW | Kupiec p | Christoffersen p | DM (IS) |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Random walk | 89.3% | 4.18 | 7.87 | 1.000 | 0.58 | 0.00 | – |
| EWMA | 90.7% | 8.87 | 13.92 | 1.770 | 0.55 | 0.00 | -5.56*** |
| HAR | 89.3% | 4.12 | 7.65 | 0.973 | 0.58 | 0.05 | +1.64 |
| AR(1)-GARCH-t | 90.2% | 4.42 | 7.32 | 0.931 | 0.84 | 0.79 | +1.63 |
| HAR-GARCH-t | 89.6% | 4.28 | 7.26 | 0.922 | 0.76 | 0.83 | +1.48 |

## 95% prediction intervals

| Model | Coverage | Avg width | Interval score | IS / RW | Kupiec p | Christoffersen p | DM (IS) |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Random walk | 94.2% | 5.57 | 10.45 | 1.000 | 0.37 | 0.00 | – |
| EWMA | 94.8% | 11.73 | 18.06 | 1.729 | 0.83 | 0.00 | -4.11*** |
| HAR | 94.5% | 5.37 | 10.18 | 0.975 | 0.58 | 0.00 | +1.30 |
| AR(1)-GARCH-t | 94.4% | 5.72 | 9.43 | 0.903 | 0.47 | 0.64 | +1.43 |
| HAR-GARCH-t | 94.8% | 5.58 | 9.38 | 0.898 | 0.83 | 0.97 | +1.28 |

## Whole-distribution score

| Model | Avg pinball loss | vs RW | DM |
| :-- | --: | --: | --: |
| Random walk | 0.4022 | 1.000 | – |
| EWMA | 0.7674 | 1.908 | -10.59*** |
| HAR | 0.3923 | 0.975 | +2.20** |
| AR(1)-GARCH-t | 0.3912 | 0.973 | +2.50** |
| HAR-GARCH-t | 0.3829 | 0.952 | +2.51** |
