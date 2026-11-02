# Monthly model monitoring

I monitored the stored predictions for November 2024 to January 2025 from `xgboost_20241031_v1`. This is a retrospective replay. Predictions were generated using application-time features, while the MOB 6 labels are only used after they become available. The monitoring step reads the stored predictions instead of scoring the applications again.

Run from the workspace root:

```bash
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_monitoring.py --snapshot-date 2024-11-01 --as-of-date 2025-05-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_monitoring.py --snapshot-date 2024-12-01 --as-of-date 2025-06-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_monitoring.py --snapshot-date 2025-01-01 --as-of-date 2025-07-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python plot_monitoring.py
```

An immature-label check with `--snapshot-date 2025-01-01 --as-of-date 2025-06-30 --dry-run` returned `performance_available=false`, `labelled_count=0`, and the performance metrics were null. Stability metrics were still calculated, and the dry run did not write to Gold.

## Stability and performance definitions

I use prediction count, mean and median predicted probability, and score PSI to monitor stability.

The PSI reference is the 2,986 May–October 2024 OOT applications. These applications are scored using the same saved model and Gold feature store. The reference score distribution is divided into ten fixed bins using reference quantiles, and the same bin boundaries are then applied to each monitored month.

PSI is calculated as: `sum((current_share − reference_share) × ln(current_share / reference_share))`

A small 1e-6 value is used to avoid issues with empty bins. Labels are not needed for PSI.

For interpretation, I use the following ranges as general guidelines:
- PSI < 0.10: low shift
- PSI 0.10–0.25: moderate shift
- PSI ≥ 0.25: larger shift that may need investigation

These ranges are used for monitoring only and do not automatically trigger retraining.

Performance is calculated only after the MOB 6 label becomes available. Once `as_of_date >= feature_snapshot_date + 6 months`, the stored predictions are joined to `label_store` by `loan_id`.

The workflow checks that the prediction and label records match, that the label date is the expected MOB 6 date, and that the label definition is `30dpd_6mob`. It then calculates actual `default rate`, `ROC-AUC`, and `Gini (2 × AUC − 1)`. Before label maturity, these fields remain null.

Monitoring results are written to: `datamart/gold/model_monitoring/`

The table contains one row per application month and model version. It stores the application month, model version, prediction count, mean and median predicted probability, score PSI, expected label date, monitoring as-of date, label availability, labelled count, actual default rate, ROC-AUC, and Gini.

Rerunning a month overwrites only that month's partition.

## Historical replay results

| Application month | Predictions / labels | Mean risk | Median risk | Score PSI | Actual default rate | ROC-AUC | Gini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-11 | 488 / 488 | 0.2776 | 0.1638 | 0.0819 | 0.2807 | 0.8088 | 0.6177 |
| 2024-12 | 515 / 515 | 0.2427 | 0.1527 | 0.0598 | 0.2718 | 0.8482 | 0.6965 |
| 2025-01 | 526 / 526 | 0.2786 | 0.1666 | 0.0654 | 0.2681 | 0.7781 | 0.5562 |

All three PSI values are below 0.10, so the score distributions show relatively low shift compared with the OOT reference.

December had the lowest mean predicted risk, but its PSI remained low. January had the weakest discrimination, with ROC-AUC decreasing to 0.7781 from 0.8482 in December. This is worth monitoring, but one weaker monthly cohort is not enough to conclude that model performance is consistently deteriorating.

The actual default rate was more stable across the three months than the mean predicted risk. These results are descriptive and are not used as a full calibration assessment.

The monitoring charts are:
- Model performance
- Predicted risk vs actual default rate
- Score PSI

The final Gold monitoring table contains three unique month/model rows. Prediction and label counts match for each mature cohort. Rerunning November kept one row for that month and did not change the December or January partitions.
