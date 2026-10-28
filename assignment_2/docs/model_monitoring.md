# Assignment 2 monthly model monitoring (Step 8)

This is a retrospective monitoring replay for the frozen `xgboost_20241031_v1` champion. Stored application-time predictions are monitored for November 2024, December 2024, and January 2025. Predictions are never regenerated here. Labels are read only when the supplied monitoring `as-of` date reaches that cohort's MOB 6 label date.

From the workspace root, run:

```bash
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_monitoring.py --snapshot-date 2024-11-01 --as-of-date 2025-05-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_monitoring.py --snapshot-date 2024-12-01 --as-of-date 2025-06-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_monitoring.py --snapshot-date 2025-01-01 --as-of-date 2025-07-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python plot_monitoring.py
```

For an immature-label check without changing Gold, use `--snapshot-date 2025-01-01 --as-of-date 2025-06-30 --dry-run`. This produced `performance_available=false`, `labelled_count=0`, and null actual default rate, ROC-AUC, and Gini, while stability measures remained available.

## Stability and performance definitions

Stability uses prediction count, mean and median predicted probability, and score PSI. The PSI reference consists of **2,986** May–October 2024 OOT applications scored afresh with the **same saved champion** using application-time Gold features only. Nine quantiles from this reference alone form ten fixed score bins; duplicate edges are removed if necessary, and the outer bins extend to infinity. Each monitored month is counted against those same boundaries. PSI is `sum((current_share − reference_share) × ln(current_share / reference_share))`, with `1e-6` protection for empty bins. No label enters PSI. For descriptive reading, PSI below 0.10 suggests low shift, 0.10–0.25 moderate shift, and 0.25 or above a shift worth investigating; these are guidelines, not automatic retraining rules.

Performance becomes available only when `as_of_date >= feature_snapshot_date + 6 months`. The workflow then joins stored predictions to `label_store` one-to-one by `loan_id`, checks the MOB 6 date and `30dpd_6mob` definition, and calculates actual default rate, probability-based ROC-AUC, and Gini (`2 × AUC − 1`). Before maturity, performance fields are null rather than zero.

The Gold table is `datamart/gold/model_monitoring/`, partitioned by `feature_snapshot_date`, at one row per month and model version. Its fields are `feature_snapshot_date` (date), `model_version` (string), `prediction_count` (int), `mean_predicted_probability` and `median_predicted_probability` (double), `score_psi` (double), `expected_label_snapshot_date` and `performance_as_of_date` (date), `performance_available` (boolean), `labelled_count` (int), and nullable `actual_default_rate`, `roc_auc`, `gini` (double). Rerunning a month replaces only that partition.

## Historical replay results

| Application month | Predictions / labels | Mean risk | Median risk | Score PSI | Actual default rate | ROC-AUC | Gini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-11 | 488 / 488 | 0.2776 | 0.1638 | 0.0819 | 0.2807 | 0.8088 | 0.6177 |
| 2024-12 | 515 / 515 | 0.2427 | 0.1527 | 0.0598 | 0.2718 | 0.8482 | 0.6965 |
| 2025-01 | 526 / 526 | 0.2786 | 0.1666 | 0.0654 | 0.2681 | 0.7781 | 0.5562 |

All three PSI values are below 0.10, so these score distributions show low shift under the stated guideline. December's lower mean score does not correspond to a large PSI shift. January has the weakest observed discrimination (AUC 0.7781 versus 0.8482 in December), which merits continued observation; three months of roughly 500 loans each do not establish a sustained performance decline. The actual default rate varies less than mean predicted risk across these months. These are descriptive comparisons, not causal or full calibration conclusions.

Charts for the final slideument are [performance](../reports/monitoring/performance.png), [predicted risk vs actual default rate](../reports/monitoring/risk_vs_actual.png), and [score PSI](../reports/monitoring/score_psi.png). The final Gold has exactly three rows, no duplicate month/version keys, valid PSI and probability-based performance fields, and matching prediction/label counts. Replaying November left its row count at one and the other two partition files unchanged. No retraining, feature selection, CSI, or automated governance action is performed.
