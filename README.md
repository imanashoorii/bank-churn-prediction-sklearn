# Bank Customer Churn Prediction

A supervised machine learning pipeline that predicts which bank customers are likely to close their accounts, so a retention team could target them with offers before they leave.

## Overview

- **Task:** Binary classification
- **Target:** `Exited` (1 = customer churned, 0 = retained)
- **Dataset:** [Bank Customer Churn (Churn Modelling)](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) — 10,000 customers, 14 features
- **Best model:** Random Forest (tuned) — **0.865 ROC-AUC**, 68% recall on churners

## Problem

A retail bank wants to identify at-risk customers early enough for the retention team to intervene. The target is imbalanced (~80% stay / ~20% churn), so the pipeline is built around recall/precision/F1/ROC-AUC rather than raw accuracy, which would be misleading on its own.

## Approach

1. **EDA** — explored target imbalance, feature distributions, and churn rate by category. The standout finding: churn rate by `NumOfProducts` is sharply non-linear (7.6% at 2 products → 82.7% at 3 → 100% at 4), a pattern linear correlation completely misses (correlation coefficient ≈ -0.05 despite being the strongest categorical signal in the data).
2. **Feature engineering** — dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`); engineered a `HasZeroBalance` flag after finding 36.2% of customers hold exactly €0 balance, with a meaningfully different churn rate (13.8% vs 24.1%).
3. **Preprocessing comparison** — compared scaled vs. unscaled numeric features across Logistic Regression and KNN. Unscaled KNN collapses to near-random performance (0.525 ROC-AUC vs. 0.779 scaled), since distance-based models are dominated by whichever feature has the largest numeric range.
4. **Modeling** — trained and compared Logistic Regression (linear baseline) against Random Forest (non-linear ensemble), inside `scikit-learn` `Pipeline`/`ColumnTransformer` objects to prevent preprocessing leakage.
5. **Tuning** — `GridSearchCV` (5-fold stratified CV, scored on F1) over both models. `class_weight="balanced"` won in every configuration tested, confirming the class imbalance was worth correcting for.
6. **Evaluation** — final, one-time check against a held-out test set (20%, stratified).

## Results

| Model | Test ROC-AUC | Test F1 | Precision (Exited) | Recall (Exited) |
|---|---|---|---|---|
| Logistic Regression (tuned) | 0.777 | 0.502 | 0.39 | 0.71 |
| **Random Forest (tuned)** | **0.865** | **0.623** | **0.57** | 0.68 |

**Top predictive features (Random Forest):** Age (0.335), NumOfProducts (0.212), Balance (0.100), EstimatedSalary (0.073), CreditScore (0.072) — consistent with patterns found independently during EDA.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`)
- matplotlib

## Project Structure

```
.
├── churn_prediction.ipynb   # Main notebook: EDA → preprocessing → modeling → evaluation
├── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

The notebook loads data directly from a public CSV URL, so no manual dataset download is required.

## Limitations

- Single static snapshot per customer, no timestamps — can't validate against a real prediction horizon or check for drift
- Only 60 customers in the dataset hold 4 products — the "100% churn" pattern at that tier is based on a very small group
- No external validation set; all results come from one train/test split of one dataset

## License

MIT