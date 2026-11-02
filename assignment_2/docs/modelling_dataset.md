# Modelling data and chronological split

## Grain, target, and loading

`utils/modelling_data.py` joins the Gold `feature_store` and `label_store` using `loan_id`, with one row representing one loan application. The joined data keeps the loan and customer IDs, feature and label dates, the target, and 71 candidate predictors.

The target is `30dpd_6mob`, defined as `DPD >= 30 at MOB 6`. The full join contains 12,500 unique loans, including 3,602 defaults, with no unmatched feature or label records.

The loader introduces an `as_of_date` and only keeps labels that would have been available by that date. This is used to avoid using future outcomes during model development. Missing feature values are left unchanged at this stage and handled later in model preprocessing. I also keep the existing Gold tables as the source of truth instead of creating another modelling-data copy.

The 71 candidate predictors consist of 67 numeric and four categorical features. The numeric features include existing application and financial fields, the four newly engineered financial features, two clickstream history indicators, and 40 three-month clickstream summary features. The categorical features are `Occupation`, `Credit_Mix`, `Payment_of_Min_Amount`, and `Payment_Behaviour`.

`MODEL_FEATURES` acts as an explicit allowlist. IDs, dates, the target, DPD, overdue and paid amounts, and post-application outcome fields are not included as predictors. I also exclude the raw `Type_of_Loan` field because it has 6,261 distinct values across 12,500 records, which would create a very high-cardinality input. The existing `Type_of_Loan_has_unknown` indicator is retained.

Financial features are matched to the application-date snapshot. Clickstream features use only the three complete months before the application date. Application dates range from `2023-01-01 to 2025-01-01`, while MOB 6 label dates range from `2023-07-01 to 2025-07-01`. Although all labels are available in the completed historical dataset, only labels known by the selected fitting date can be used for model development.

## Monthly application profile

| Application month | Loans | Defaults | Default rate | MOB 6 label month |
| --- | ---: | ---: | ---: | --- |
| 2023-01 | 530 | 140 | 26.4% | 2023-07 |
| 2023-02 | 501 | 154 | 30.7% | 2023-08 |
| 2023-03 | 506 | 162 | 32.0% | 2023-09 |
| 2023-04 | 510 | 137 | 26.9% | 2023-10 |
| 2023-05 | 521 | 139 | 26.7% | 2023-11 |
| 2023-06 | 517 | 137 | 26.5% | 2023-12 |
| 2023-07 | 471 | 140 | 29.7% | 2024-01 |
| 2023-08 | 481 | 139 | 28.9% | 2024-02 |
| 2023-09 | 454 | 119 | 26.2% | 2024-03 |
| 2023-10 | 487 | 125 | 25.7% | 2024-04 |
| 2023-11 | 491 | 143 | 29.1% | 2024-05 |
| 2023-12 | 489 | 151 | 30.9% | 2024-06 |
| 2024-01 | 485 | 144 | 29.7% | 2024-07 |
| 2024-02 | 518 | 147 | 28.4% | 2024-08 |
| 2024-03 | 511 | 168 | 32.9% | 2024-09 |
| 2024-04 | 513 | 142 | 27.7% | 2024-10 |
| 2024-05 | 491 | 149 | 30.3% | 2024-11 |
| 2024-06 | 498 | 155 | 31.1% | 2024-12 |
| 2024-07 | 505 | 155 | 30.7% | 2025-01 |
| 2024-08 | 543 | 163 | 30.0% | 2025-02 |
| 2024-09 | 493 | 163 | 33.1% | 2025-03 |
| 2024-10 | 456 | 112 | 24.6% | 2025-04 |
| 2024-11 | 488 | 137 | 28.1% | 2025-05 |
| 2024-12 | 515 | 140 | 27.2% | 2025-06 |
| 2025-01 | 526 | 141 | 26.8% | 2025-07 |

## Chronological partitions

I use **2024-10-31** as the model fitting cut-off. By this date, all Train and Validation labels are available.

The data is split chronologically. Train uses 2023 applications, followed by Validation from January to April 2024. Applications from May to October 2024 are kept as a retrospective OOT set. Their final MOB 6 labels become available in April 2025, so the complete OOT period can only be evaluated after that point.

Applications from November 2024 to January 2025 are kept separately for the later monthly inference and monitoring exercise using the frozen champion model.

| Partition | Application months | Loans | Defaults | Default rate | Latest label date | Treatment |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Train | 2023-01 to 2023-12 | 5,958 | 1,686 | 28.30% | 2024-06-01 | Used to fit candidate models. |
| Validation | 2024-01 to 2024-04 | 2,027 | 601 | 29.65% | 2024-10-01 | Used to compare models and select the champion. |
| Retrospective OOT | 2024-05 to 2024-10 | 2,986 | 897 | 30.04% | 2025-04-01 | Held out from model fitting and selection; used as a later generalisation check. |
| Future inference / monitoring cohort | 2024-11 to 2025-01 | 1,529 | 418 | 27.34% | 2025-07-01 | Reserved for month-by-month inference and monitoring with the frozen model. |

The four periods together cover all 12,500 loans. Train and Validation are used for model development, while the OOT period is kept separate from model selection. The November 2024 to January 2025 applications are not used during model development and are reserved for the later inference and monitoring steps.

The default rates shown for the OOT and later inference cohorts are retrospective statistics from the completed dataset. They were not available at the original fitting date and are not used as model-development inputs.

`feature_snapshot_date` is the loan application date, while `label_snapshot_date` is the date when the MOB 6 outcome becomes available. `as_of_date` is used to control which labels can be considered known at a given point in time.

The three inference cohorts receive their MOB 6 labels in May, June, and July 2025. The OOT result is also only fully available later, after its final label matures in April 2025. For this reason, the later scoring and monitoring exercise is treated as a retrospective replay.

## Missingness in the full joined data

| Candidate predictors | Missing rows | Share |
| --- | ---: | ---: |
| `Age` | 319 | 2.55% |
| `Num_Bank_Accounts` | 4 | 0.03% |
| `Num_of_Loan` | 500 | 4.00% |
| `Num_of_Delayed_Payment` | 84 | 0.67% |
| `Changed_Credit_Limit` | 254 | 2.03% |
| `Amount_invested_monthly` | 558 | 4.46% |
| `Monthly_Balance` | 1 | 0.01% |
| `num_fin_products` | 504 | 4.03% |
| Each of the 40 clickstream mean/latest columns | 4,056 | 32.45% |
| Other three new financial features, four categorical fields, and `click_history_months` / `click_history_missing` | 0 | 0% |

The 4,056 applications without eligible clickstream history are kept in the dataset. These rows are identified by `click_history_missing` = 1, while their 40 clickstream summary fields remain null.

`num_fin_products` also remains null when one of its source counts is missing. No missing values is filled in the Gold data or in this loading step. Imputation is handled later inside the modelling pipeline and is fitted only on the appropriate training rows.
