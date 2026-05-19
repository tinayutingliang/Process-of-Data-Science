# NYC 311 Service Requests TimeSeries Forecasting

## Problem
Predict daily service requests to optimize staffing under asymmetric cost (under‑staffing costs twice as much as over‑staffing).

## Methods
- Baseline: weekday mean
- Ridge regression with lag features (lag 7, 14, 21, 28)
- HistGradientBoosting with recency‑weighted same‑day lags

## Evaluation Metrics
- MAE (mean absolute error)
- Business loss (custom asymmetric cost function)

## Results
- Reduced business loss by ~48.7% compared to baseline
- Better tracking of seasonal descent (Nov–Dec)

## Files
- `NYC311_Forecasting_Report.pdf` – detailed analysis and model comparison
- `NYC311_Forecasting.ipynb` – feature engineering, modeling, evaluation
