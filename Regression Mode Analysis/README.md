# Mode‑Wise Regression Analysis

## Problem
Build a predictive model for a continuous target `y` using numeric and categorical features.  
The relationship between `x` and `y` varies significantly across different `mode` values, requiring a **mode‑wise modeling strategy**.

## Approach
- Data preprocessing: remove `row_id`, `note`; convert timestamp → hour/day/weekday/month
- Visual EDA reveals distinct patterns per mode
- **Mode‑specific models**:
  - Mode 1 & 4 → Linear Regression
  - Mode 2 → Polynomial Regression (degree 5)
  - Mode 3 → KMeans (2 clusters) + cluster‑mean prediction
  - Mode 5 → Polynomial Regression (degree 2)
  - Mode 6 → split by category, separate Linear Regression

## Key Results
| Mode | RMSE |
|------|------|
| 1    | 1.01 |
| 2    | 1.21 |
| 3    | 1.01 |
| 4    | 1.07 |
| 5    | 2.19 |
| 6    | 1.79 |

## Key Takeaways
- **Heterogeneous data benefits from local models** rather than a single global model.
- For Mode 3, cluster‑mean outperformed cluster + linear regression (lower validation RMSE), showing that simpler can be better.
- Residual analysis shows higher variance in noisier modes (5 & 6).

## Files
- `Regression_Report.pdf` – full report with EDA, model selection, residual diagnostics
- `Regression.py` – complete pipeline: preprocessing → EDA → training → validation → test prediction
