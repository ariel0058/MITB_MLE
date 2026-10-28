# Assignment 2 Airflow pipeline

`assignment2_monthly_pipeline` runs the existing scripts in order:

`prepare_datamart` (`main.py --stage all`) → `build_model_bank` (`model_train.py --save-model`) → `monthly_inference` → `monthly_monitoring`.

The DAG is scheduled for the first day of each month (`0 0 1 * *`, UTC), with catchup enabled, start date 2024-11-01, end date 2025-01-01, and `max_active_runs=1`. Its three logical application dates are 2024-11-01, 2024-12-01 and 2025-01-01. Monthly inference receives `{{ ds }}`. Monitoring receives `{{ ds }}` plus six months as its label-knowledge date (2025-05-01, 2025-06-01, 2025-07-01). The latter dates are when MOB 6 labels can first be assessed. This is a retrospective coursework replay, not a claim of historical deployment.

The first two tasks reproduce shared inputs and the same deterministic `xgboost_20241031_v1` artefact on each run. They do not expand the fitting period or create a monthly model version. Monthly tasks consume the frozen model. Sequential DAG runs and month-level overwrite behaviour make retries and backfill safe. Plot generation remains a separate command after the three monitoring rows exist.

From `assignment_2/`:

```sh
docker compose build
docker compose up -d
docker compose exec -T airflow airflow dags list-import-errors
docker compose exec -T airflow airflow dags backfill \
  -s 2024-11-01 -e 2025-01-01 assignment2_monthly_pipeline
```

Airflow UI: http://127.0.0.1:8080/ . Log in as `admin`; the generated password is in `/opt/airflow/standalone_admin_password.txt` (read it with `docker compose exec -T airflow cat /opt/airflow/standalone_admin_password.txt`). The DAG is initially paused; the CLI backfill command above runs the three historical dates. It can be unpaused in the UI for scheduled catchup instead. Check each task in the three DAG runs. The Step 0 `assignment2_smoke_test` DAG only checks the Airflow environment.

The Airflow container runs scripts directly from the bind-mounted `/app` directory. Outputs persist in `datamart/bronze/`, `datamart/silver/`, `datamart/gold/`, `model_bank/xgboost_20241031_v1/` and `reports/monitoring/` under `assignment_2/`. Expected monthly prediction counts are 488, 515 and 526; the monitoring table has one row per month and model version.

## Validated historical replay

The three backfill DAG runs and all 12 core task instances succeeded. The rebuilt feature and label stores each contain 12,500 unique loans. The model bank contains `model.joblib` and `metadata.json` for `xgboost_20241031_v1` (7,985 training rows; final OOT AUC 0.8203, Gini 0.6406). The prediction table contains 1,529 unique loans, with 488/515/526 in Nov/Dec/Jan. Monitoring contains exactly three unique month/model rows:

| Application month | Score PSI | ROC-AUC | Gini |
| --- | ---: | ---: | ---: |
| 2024-11-01 | 0.081905 | 0.808836 | 0.617672 |
| 2024-12-01 | 0.059811 | 0.848238 | 0.696476 |
| 2025-01-01 | 0.065352 | 0.778088 | 0.556176 |

`airflow tasks test assignment2_monthly_pipeline monthly_inference 2024-11-01` and the corresponding `monthly_monitoring` task were rerun. Prediction counts stayed at 1,529 total/488 for Nov and monitoring stayed at three rows/one for Nov. The Step 8 date guard rejects an as-of date before the application month; an as-of date before label maturity keeps performance fields null in dry-run mode.
