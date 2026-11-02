# Model governance and deployment

## Current model and evidence

The current champion is `xgboost_20241031_v1`. It was fitted using applications from `January 2023 to April 2024`, with labels available by the `2024-10-31` fitting cut-off. The target is `30dpd_6mob`, which indicates whether a loan reaches at least 30 DPD at MOB 6.

The final model achieved the following results on the May–October 2024 retrospective OOT cohort. The OOT results were only used after model selection and final refitting.
- ROC-AUC: **0.8203**
- Gini: **0.6406**

| Application month | Score PSI | ROC-AUC | Gini |
| --- | ---: | ---: | ---: |
| 2024-11 | 0.0819 | 0.8088 | 0.6177 |
| 2024-12 | 0.0598 | 0.8482 | 0.6965 |
| 2025-01 | 0.0654 | 0.7781 | 0.5562 |

These are retrospective monthly monitoring results, not evidence that the model was deployed at the time.

## Monthly monitoring and review triggers

Both model stability and performance are monitored.

For stability, I review prediction volume, mean and median predicted risk, score PSI, and input data quality. Schema changes, missing required fields, or other input-quality failures should be investigated immediately. I use the following PSI ranges as monitoring guidelines:

| Score PSI | Response |
| --- | --- |
| < 0.10 | Low shift; continue monitoring. |
| 0.10 to < 0.25 | Investigate the customer mix and score distribution. |
| ≥ 0.25 | Review the model and the changed population. |

These ranges are used as signals for investigation rather than automatic retraining rules. Performance can only be reviewed once the MOB 6 labels become available. At that point, I monitor actual default rate, ROC-AUC, and Gini.

A model refresh should be considered when performance deteriorates materially across several mature cohorts, especially if this happens together with population shift or a business change. Other reasons for review include:
- Major changes in lending policy or customer mix
- Changes to important features or schemas
- Persistent data-quality issues
- Enough newly matured labelled data becoming available for a scheduled model review

A single weaker month is not enough to trigger retraining. January 2025 has the lowest AUC at 0.7781, so we should continue monitor it, but there is not yet evidence of a sustained decline.

## Model refresh process

If a model review is triggered, I would follow this process:
Monitor → Review trigger → Prepare newly matured data → Train challenger → Validate and select → Refit on Train + Validation → Evaluate once on OOT → Compare with current champion → Human approval → Save new version → Deploy → Monitor

The challenger should only use labels that are available at its fitting date.

Preprocessing such as imputation and encoding should be fitted on the challenger training data only. Validation is then used to compare candidate models or configurations.

After selecting the challenger configuration, I would refit a fresh pipeline on Train + Validation. The refitted challenger would then be evaluated once on a separate chronological OOT period.

The OOT period should remain independent and should not be repeatedly used for tuning. The challenger can then be compared with the current champion using model performance, stability, data quality, and operational suitability.

The current champion remains in use until the challenger passes these checks and is approved for promotion.

## Versioning and rollback

The current model bank contains:

```
model_bank/
└── xgboost_20241031_v1/
    ├── model.joblib
    └── metadata.json
```

A future approved refresh should be stored as a new model version instead of overwriting the existing champion. For example:

```text
model_bank/
├── xgboost_20241031_v1/
│   ├── model.joblib
│   └── metadata.json
└── future_model_v2/          # example only, not created in this assignment
    ├── model.joblib
    └── metadata.json
```

Each version should keep its fitted preprocessing, model, feature order, training cut-off, and evaluation results together.

Previous approved versions should remain available for reproducibility, comparison, audit, and rollback. If a newly deployed model produces unexpected scores, data issues, or operational problems, the previous approved version can be restored while the issue is investigated.

## Deployment approach

Monthly batch inference is the most suitable deployment option for this project. The application and monitoring data are already organised by monthly cohorts, and there is no requirement for an immediate real-time loan decision. Airflow also supports the monthly schedule, retry, and backfill process used in the pipeline Using batch inference keeps the implementation simple and allows each monthly cohort to be scored with the same frozen pipeline and reviewed consistently.

Real-time inference could be considered later if the business requires instant decisions and all required input features can be produced reliably at decision time.

## Current decision

All three monitored PSI values are below 0.10, so there is no strong evidence of score distribution shift.

The model continues to show useful discrimination across the three monitored months. January 2025 is weaker than November and December, but one lower monthly AUC is not enough to conclude that the model is consistently deteriorating.

`xgboost_20241031_v1` remains the current champion. No immediate retraining is required, and monthly monitoring should continue.