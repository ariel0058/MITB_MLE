# Monthly inference

The saved `xgboost_20241031_v1` model is used to score three later application months: November 2024, December 2024, and January 2025. This is a retrospective replay of monthly inference rather than a historical deployment.

Inference uses the Gold feature store together with the saved fitted pipeline. Since the target is based on MOB 6 outcomes, labels are not available at application time and are not used in this step.

Run one month at a time from the workspace root:

```bash
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_inference.py --snapshot-date 2024-11-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_inference.py --snapshot-date 2024-12-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_inference.py --snapshot-date 2025-01-01
```

`utils.model_store.load_model` loads the fitted pipeline and metadata. The script checks that the required input features are present with the expected data types and order, then passes them to `predict_proba`. Only application months after the 2024-10-31 fitting cut-off are accepted.

No preprocessing or model fitting is repeated during inference. The saved imputation, encoding, and XGBoost model are reused directly.

Predictions are written to: `datamart/gold/model_predictions/`

The table is partitioned by feature_snapshot_date and contains:
- loan_id
- Customer_ID
- feature_snapshot_date
- model_version
- predicted_default_probability

No label or future outcome fields are included. Rerunning a month overwrites only that month's partition.

| Application month | Predictions | Unique loans | Min probability | Max probability | Mean probability | Median probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-11-01 | 488 | 488 | 0.021580 | 0.731027 | 0.277598 | 0.163750 |
| 2024-12-01 | 515 | 515 | 0.016178 | 0.732174 | 0.242681 | 0.152694 |
| 2025-01-01 | 526 | 526 | 0.022250 | 0.735901 | 0.278643 | 0.166584 |

Across the three months, 1,529 eligible applications produced 1,529 predictions. There were no duplicate (loan_id, model_version) keys, null IDs, or null scores, and all probabilities were within [0, 1].

November is reran to check the monthly overwrite logic. It remained at 488 rows, while the December and January partitions were unchanged.

These results describe the prediction score distribution only. Model performance is evaluated later, once the corresponding MOB 6 labels become available.
