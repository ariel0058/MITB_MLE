# Assignment 2 modelling data (Step 4)

## Grain, target, and loading

`utils/modelling_data.py` joins Gold `feature_store` and `label_store` by `loan_id`. The grain is one loan application. It retains `loan_id`, `Customer_ID`, `feature_snapshot_date`, `label_snapshot_date`, `label`, and an explicit list of 71 candidate predictors. The target is `30dpd_6mob`: DPD >= 30 at MOB 6. The full joined data has 12,500 unique loans, 3,602 defaults, and no unmatched rows. Missing predictors remain missing; no imputation or preprocessing is performed here. The reusable loader requires `as_of_date` and returns only rows with `label_snapshot_date <= as_of_date`. There is no extra Parquet copy of these existing Gold tables.

The 71 predictors comprise 67 numeric fields (21 existing application/financial fields, four Step 3 financial ratios/counts, two clickstream history indicators, and 40 three-month clickstream summaries) plus four categorical fields: `Occupation`, `Credit_Mix`, `Payment_of_Min_Amount`, and `Payment_Behaviour`. `MODEL_FEATURES` is an explicit allowlist; IDs, dates, target, DPD, overdue/paid amounts, and post-application balances cannot enter through it. The raw `Type_of_Loan` string is retained in the underlying Gold store but excluded from the candidate list because it has 6,261 distinct values in 12,500 rows; its existing `Type_of_Loan_has_unknown` indicator remains available. No encoding decision or model fitting is made in Step 4.

All selected non-clickstream predictors come from the application-time Gold row. Financial inputs are matched on `(Customer_ID, feature_snapshot_date)`; clickstream summaries use the three complete months before application. `feature_snapshot_date` spans 2023-01-01 to 2025-01-01, and `label_snapshot_date` spans 2023-07-01 to 2025-07-01. Every label date is exactly six months after application. A stored label is **not** automatically eligible at an earlier model fitting date.

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

## Proposed chronological design

The proposed model fitting cut-off remains **2024-10-31**. Train and Validation labels are mature by then. The May–October 2024 cohort is reserved for **retrospective chronological OOT evaluation**: its applications are later than Train and Validation, but its labels are not all observable at the fitting cut-off. Its last MOB 6 label is dated 2025-04-01, so the full OOT period can be evaluated as of **2025-04-30**. OOT rows must not enter fitting, tuning, or champion selection. The November 2024–January 2025 cohort is reserved for a separate monthly inference/monitoring demonstration using a frozen champion; it must not enter model development or OOT evaluation used for model selection.

| Partition | Application months | Loans | Defaults | Default rate | Latest label date | Treatment |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Train | 2023-01 to 2023-12 | 5,958 | 1,686 | 28.30% | 2024-06-01 | Mature at fitting cut-off. |
| Validation | 2024-01 to 2024-04 | 2,027 | 601 | 29.65% | 2024-10-01 | Mature at fitting cut-off. |
| Retrospective OOT | 2024-05 to 2024-10 | 2,986 | 897 | 30.04% | 2025-04-01 | Hold out from fitting and tuning; evaluate after MOB 6 labels mature. |
| Future inference / monitoring cohort | 2024-11 to 2025-01 | 1,529 | 418 | 27.34% | 2025-07-01 | Score month by month with a frozen champion; reserve entirely from model development and OOT model-selection evaluation. |

The four periods cover all 12,500 rows without overlap or discarded records. Train, Validation, and OOT each contain both label classes. Model development uses Train → Validation → champion selection. The frozen champion is then replayed on the reserved November 2024–January 2025 cohorts for monthly inference and monitoring. The May–October 2024 OOT cohort is evaluated separately after its MOB 6 labels mature and is not used for champion selection. OOT's reported default rate is a retrospective profile, not information available at the fitting cut-off. Do not claim that this model was deployed and scored the OOT applications in real time.

`feature_snapshot_date` is the application date when predictors are available; `label_snapshot_date` is the MOB 6 outcome date; `as_of_date` is the simulated knowledge date controlling label eligibility. The proposed frozen model could score the reserved cohorts using application-time features, but performance becomes available only after their labels mature: 2024-11 → 2025-05, 2024-12 → 2025-06, and 2025-01 → 2025-07. Because the full OOT result only becomes available in April 2025, it cannot be used to approve a champion before scoring those earlier reserved months. Any later replay of their monthly scores must be labelled as a retrospective demonstration, not as a claim of historical deployment.

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

The 4,056 loans without clickstream history remain present, identified by `click_history_missing = 1`; their 40 summary fields stay null. `num_fin_products` stays null when any required count is missing. Missing-value treatment belongs to Step 5 and must be learned from Train only.
