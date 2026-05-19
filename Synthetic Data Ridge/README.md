# Ridge Regression on Synthetic sin(x) and sin(10x)

## Goal
Demonstrate how polynomial features affect Ridge regression’s ability to fit non‑linear functions.

## Key Experiments
- Fit Ridge on `y = sin(x)` with only `x` → high bias
- Add polynomial features `x² … x⁵` → MAE drops to near noise level
- Replace with `y = sin(10x)` → polynomial features fail (high frequency)

## Why This Matters
Shows that **feature representation** matters as much as model choice, and that low‑degree polynomials cannot capture rapid oscillations.

## Files
- `polynomial_features.ipynb` – full code, visualizations, and explanation
