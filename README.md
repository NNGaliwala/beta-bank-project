# beta-bank-project

## Overview
Beta Bank is losing customers gradually each month. This project builds a binary classification model to predict customer churn, allowing the bank to proactively target at-risk customers with retention offers before they leave.

**Goal:** Achieve an F1 score ≥ 0.59 on the held-out test set.

## Dataset
- 10,000 customer records with 14 features including credit score, geography, gender, age, tenure, balance, number of products, and activity status
- Target variable: `Exited` (1 = churned, 0 = stayed)
- Class imbalance: ~80% non-churners, ~20% churners

## Approach

### Preprocessing
- Filled ~900 missing `Tenure` values with the median
- Dropped non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
- One-hot encoded `Geography` and `Gender` using `pd.get_dummies`
- Split data 60/20/20 into train, validation, and test sets

### Models Evaluated
| Model | Validation F1 |
|---|---|
| Logistic Regression (baseline) | 0.088 |
| Decision Tree (baseline) | 0.504 |
| Random Forest (baseline) | 0.575 |

Random Forest was selected as the primary model due to its strong baseline performance with binary/categorical features.

### Handling Class Imbalance
Three approaches were tested:
1. **Class weights (`class_weight='balanced'`)** — F1 decreased to 0.564; not selected
2. **Upsampling minority class** — repeat=4 achieved F1 = 0.604; selected
3. **Threshold tuning** — lowering the decision threshold from 0.5 to 0.4 improved F1 to 0.614

### Hyperparameter Tuning
Grid search over `n_estimators` (50, 100, 150) and `max_depth` (None, 10, 15, 20) identified the best configuration: `n_estimators=50, max_depth=20`.

## Final Results
| Metric | Validation | Test |
|---|---|---|
| F1 Score | 0.620 | 0.611 |
| AUC-ROC | 0.839 | 0.842 |

The final model exceeds the required F1 threshold of 0.59 and generalizes well, with test metrics closely matching validation — indicating no overfitting during tuning.

## Tech Stack
- Python, Pandas, NumPy, Matplotlib
- Scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
