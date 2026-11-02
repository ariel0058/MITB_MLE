# Airflow pipeline

`assignment2_monthly_pipeline` calls the scripts used in the earlier stages:

`prepare_datamart` (`main.py --stage all`) → `build_model_bank` (`model_train.py --save-model`) → `monthly_inference` → `monthly_monitoring`.

The DAG uses a first-of-month schedule (`0 0 1 * *`, UTC), catchup, and `max_active_runs=1`. With a start date of 2024-11-01 and end date of 2025-01-01, Airflow creates logical dates for November, December, and January only. Inference receives that logical date as `{{ ds }}`. Monitoring receives the application month plus six months, when its MOB 6 label becomes available: 2025-05-01, 2025-06-01, and 2025-07-01. These runs replay historical cohorts; they do not represent a deployment at those dates.

For a clean start, the first two tasks rebuild the datamart and the same `xgboost_20241031_v1` artefact on each run. The fitting period and model version stay fixed; monthly inference loads that saved model. Runs execute sequentially, and the monthly outputs replace their own partitions on retry. Charts are generated separately once all monitoring rows are present.

From `assignment_2/`:

```sh
docker compose build
docker compose up -d
docker compose exec -T airflow airflow dags list-import-errors
docker compose exec -T airflow airflow dags backfill \
  -s 2024-11-01 -e 2025-01-01 assignment2_monthly_pipeline
```

The UI is at http://127.0.0.1:8080/. Log in as `admin`; retrieve the generated password with `docker compose exec -T airflow cat /opt/airflow/standalone_admin_password.txt`. The DAG starts paused, but the CLI backfill command runs the three dates without unpausing it. `assignment2_smoke_test` is a separate environment check.

The Airflow container runs the scripts from the bind-mounted `/app` directory. The `datamart/` and `model_bank/` outputs therefore persist under `assignment_2/` on the host. Plot files in `reports/monitoring/` come from the separate plotting script.

## Validated historical replay

All three backfill runs succeeded, with 12 successful task instances and no DAG import errors. The rebuilt feature and label stores each contain 12,500 unique loans. The model bank contains `model.joblib` and `metadata.json`; predictions cover 1,529 unique loans (488/515/526 by month), and monitoring has three month/model rows. The monthly PSI and AUC values are recorded in [model_monitoring.md](model_monitoring.md).

I reran the November inference and monitoring tasks with `airflow tasks test`. The prediction table still had 1,529 rows overall and 488 for November; monitoring still had three rows overall and one for November.
