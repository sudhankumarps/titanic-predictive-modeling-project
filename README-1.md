# Predictive Modeling Using Machine Learning (Titanic Survival)
**Internship Task 2 - Data Science Domain**

## Objective
Build and evaluate machine learning models that predict an outcome from data, and visualize performance with confusion matrices and ROC curves.

## Problem
Predict whether a **Titanic passenger survived**, using class, sex, age, family size, fare, and port of embarkation (891 passengers, seaborn's built-in Titanic dataset).

## Models compared
Logistic Regression, Decision Tree, Random Forest (tuned with GridSearchCV).

## Results (test set, 179 passengers)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8427 |
| Decision Tree | 0.7765 | 0.7544 | 0.6232 | 0.6825 | 0.7927 |
| Random Forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8345 |

5-fold cross-validation keeps all three models within a few points of each other (79-81%), so the results are stable. Top predictors (tuned Random Forest): fare, sex, and age.

## Files
| File | Description |
|---|---|
| `Predictive_Modeling_Titanic.ipynb` | Full executed notebook (EDA, preprocessing pipeline, training, evaluation, tuning) |
| `Predictive_Modeling_Titanic_Report.html` | Static export of the notebook, viewable in any browser |
| `Final_Report_Titanic.docx` | Written report |
| `model_results.csv` | Metrics table |
| `charts/` | Class distribution, survival breakdowns, confusion matrices, ROC curves, feature importance |

## How to run
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Predictive_Modeling_Titanic.ipynb
```
The dataset loads directly via `seaborn.load_dataset('titanic')`, so no manual download is needed.

## Tools
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
