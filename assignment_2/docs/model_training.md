# Assignment 2 model development (Step 5)

Run `python model_train.py` from `assignment_2` using the pinned Docker environment. No model artefact, prediction table, or monitoring table is written in this step.

## Data and preprocessing

Train applications: 2023-01 to 2023-12 (5,958); Validation: 2024-01 to 2024-04 (2,027). Both are loaded with `as_of_date="2024-10-31"`. OOT: 2024-05 to 2024-10 (2,986), loaded only after champion selection with `as_of_date="2025-04-30"`. Applications from 2024-11 to 2025-01 are reserved and not read into any Step 5 partition.

All 71 approved candidate fields enter each model: 67 numeric and four categorical. The numeric pipeline uses median imputation. Logistic Regression additionally uses `StandardScaler`; XGBoost does not scale numeric fields. The categorical pipeline uses most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`. Each complete pipeline learns preprocessing from its own fitting rows: Train for candidate comparison, then Train + Validation for the final XGBoost refit. Both phases receive 97 columns after one-hot expansion. Unknown values are not replaced with literal zero. The existing `click_history_missing` and `click_history_months` indicators remain alongside imputed clickstream summaries. No Gold values are changed or rows dropped for missing predictors.

Fitting was checked against the appropriate rows' medians and categorical values, plus Logistic Regression's Train scaler means. Validation uses the Train-fitted candidate pipelines only for `predict_proba`; OOT uses the final refitted pipeline only for `predict_proba`. No sampling, class weights, feature selection, threshold tuning, or hyperparameter search was used.

## Candidate comparison and champion

Both candidates were fitted once on Train. Logistic Regression uses L2 regularization (`C=1`, the sklearn default; `max_iter=2000`). XGBoost uses 160 trees, maximum depth 3, learning rate 0.05, subsample/column sample 0.9, and L2 regularization 5. The primary metric is ROC-AUC; Gini = 2 × AUC − 1.

| Candidate | Train ROC-AUC | Train Gini | Validation ROC-AUC | Validation Gini | Train − Validation AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8156 | 0.6311 | 0.8116 | 0.6232 | 0.0040 |
| XGBoost | 0.8923 | 0.7845 | 0.8409 | 0.6817 | 0.0514 |

**Selected algorithm: XGBoost.** Its Validation ROC-AUC exceeds Logistic Regression by 0.0293. The 0.0514 Train–Validation gap warns of some overfitting, but selection uses Validation ROC-AUC only. These are candidate metrics from Train-only fitted pipelines; they are not metrics for the later final refit. This decision was made before loading OOT and is not revised using OOT performance. Neither candidate fit produced a preprocessing or convergence warning.

## Final champion refit

A fresh XGBoost pipeline with the selected, unchanged configuration is refitted on **Train + Validation: 7,985 applications from 2023-01 through 2024-04**, with 2,287 defaults. All labels are mature by the 2024-10-31 fitting cut-off. The entire pipeline relearns numeric medians, categorical modes/categories, and XGBoost trees from these rows; 43 of 67 numeric medians differ from the Train-only pipeline. The final fit has 97 transformed columns and produced no warnings. No OOT or reserved future-cohort row enters this fit. The final pipeline remains in memory in Step 5; formal model-bank persistence belongs to Step 6.

## Final independent retrospective OOT check

After the refit, the final XGBoost champion scored 2,986 May–October 2024 applications, with 897 defaults (30.04%). **Final OOT ROC-AUC = 0.8203; Gini = 0.6406.** The earlier Train-only XGBoost pipeline scored 0.8179 AUC on this same OOT cohort; the final refit is higher by 0.0023. The final OOT AUC is lower than the 0.8409 Validation AUC observed during candidate selection, but those metrics come from different fitted pipelines. They are only directional evidence that future performance should be monitored, not a direct performance-drop estimate. OOT labels were not all known at the 2024-10-31 fitting date, so this is a retrospective chronological evaluation, not a claim of historical deployment. OOT did not influence selection or fitting. The 2024-11 to 2025-01 cohorts remain untouched for later monthly inference and monitoring.

## Interpretation

The **final refitted XGBoost** pipeline's ten largest transformed-column `feature_importances_` values are:

| Rank | Transformed predictor | Importance |
| ---: | --- | ---: |
| 1 | `Outstanding_Debt` | 0.1475 |
| 2 | `Credit_Mix_Bad` | 0.0542 |
| 3 | `Interest_Rate` | 0.0524 |
| 4 | `Credit_Mix_Good` | 0.0437 |
| 5 | `num_fin_products` | 0.0396 |
| 6 | `Num_Bank_Accounts` | 0.0348 |
| 7 | `Delay_from_due_date` | 0.0323 |
| 8 | `Changed_Credit_Limit` | 0.0233 |
| 9 | `Num_Credit_Inquiries` | 0.0210 |
| 10 | `fe_5_mean_3m` | 0.0192 |

The Logistic Regression baseline's largest absolute coefficients include `Payment_Behaviour_High_spent_Large_value_payments` (−0.758), `Payment_of_Min_Amount_No` (−0.749), `Payment_Behaviour_Low_spent_Small_value_payments` (+0.415), `Changed_Credit_Limit` (−0.399), and `Delay_from_due_date` (+0.393). Coefficients describe conditional associations in this fitted model, not causal effects; numeric fields were standardized, while one-hot indicators were not. Related raw and engineered financial fields may divide importance among themselves. No features are removed based on these rankings.
