# Model development

`python model_train.py` runs the model comparison and final refit. Adding **--save-model** also saves the final champion to the model bank.

## Data and preprocessing

Train contains 5,958 applications from 2023, while Validation contains 2,027 applications from January to April 2024. All labels in these two sets are available by the `fitting cut-off of 2024-10-31`.

The `May-October 2024 cohort` contains 2,986 applications and is kept as a retrospective OOT set. Its full set of MOB 6 labels is available by `2025-04-30`. Applications from `November 2024 to January 2025` are reserved for later inference and monitoring.

Both models use the same 71 candidate features: 67 numeric and four categorical. Numeric missing values are filled using the **median of the fitting data**. Logistic Regression also applies **StandardScaler**. Categorical features use **most-frequent** imputation followed by **OneHotEncoder**(handle_unknown="ignore"). After encoding, both pipelines contain 97 transformed features.

For candidate comparison, preprocessing is fitted on Train only. The final XGBoost pipeline is later refitted on Train + Validation. Missing values are not treated as zero, and applications without clickstream history are kept using `click_history_missing` and `click_history_months`.

I checked that the fitted medians, categorical values, and Logistic Regression scaling parameters came from the correct fitting rows. Validation and OOT are only transformed and scored with `predict_proba`. 

## Candidate comparison and champion

Two models were compared:
- `Logistic Regression` with L2 regularisation (C=1, max_iter=2000)
- `XGBoost` with 160 trees, maximum depth 3, learning rate 0.05, subsample and column sample of 0.9, and L2 regularisation of 5

`ROC-AUC` is the main comparison metric. `Gini` is calculated as 2 × AUC − 1.

| Candidate | Train ROC-AUC | Train Gini | Validation ROC-AUC | Validation Gini | Train − Validation AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8156 | 0.6311 | 0.8116 | 0.6232 | 0.0040 |
| XGBoost | 0.8923 | 0.7845 | 0.8409 | 0.6817 | 0.0514 |

**XGBoost was selected** because it achieved the higher Validation AUC, outperforming Logistic Regression by 0.0293.

The Train–Validation gap for XGBoost is 0.0514, which suggests some overfitting. However, its Validation performance is still stronger than Logistic Regression. The selection decision is based on Validation only, and OOT is not used to choose the model.

## Final champion refit

After selecting XGBoost, a fresh pipeline was refitted on `Train + Validation`.

The final fitting data contains 7,985 applications from January 2023 to April 2024, including 2,287 defaults. All labels in this period were available by 2024-10-31.

The complete pipeline is refitted, including numeric medians, categorical mappings, and XGBoost trees. The final pipeline still contains 97 transformed features. OOT and the later inference cohorts are not used in this fit.

## Final independent retrospective OOT check

The final XGBoost pipeline was then evaluated on the 2,986 May–October 2024 OOT applications, including 897 defaults (30.04%).
- **ROC-AUC: 0.8203**
- **Gini: 0.6406**

For comparison, the earlier Train-only XGBoost pipeline achieved an OOT AUC of 0.8179.

The final OOT AUC is lower than the 0.8409 Validation AUC observed during model selection, but these results come from different fitted pipelines and should not be interpreted as a direct performance drop.

The OOT labels were not fully available at the original fitting cut-off, so this is a retrospective evaluation. The OOT result was not used to retune or change the selected model.

## Feature importance

The ten largest `feature_importances_ values` from the final refitted XGBoost model are:

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

The Logistic Regression baseline also provides coefficient-based interpretation. Its larger absolute coefficients include `Payment_Behaviour_High_spent_Large_value_payments`, `Payment_of_Min_Amount_No`, `Payment_Behaviour_Low_spent_Small_value_payments`, `Changed_Credit_Limit`, and `Delay_from_due_date`.

These values are used for interpretation only. They do not imply causal relationships.
