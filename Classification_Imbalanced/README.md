# Medical Screening: Imbalanced Classification

## Scenario
Predict whether a patient is sick (5% prevalence). Accuracy is misleading; recall matters more.

## What This Project Shows
- Why accuracy fails in imbalanced data
- Confusion matrix, precision, recall interpretation
- How lowering the threshold increases recall at the cost of precision

## Key Insight
For medical screening, **recall > precision** because missing a sick patient is far worse than a false alarm.

## Files
- `imbalanced_classification.ipynb`
