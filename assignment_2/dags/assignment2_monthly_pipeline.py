"""Run the data, model, prediction and monitoring scripts each month."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="assignment2_monthly_pipeline",
    description="Reproduce the datamart, frozen champion, inference and monitoring",
    schedule="0 0 1 * *",
    start_date=datetime(2024, 11, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["assignment2", "monthly"],
) as dag:
    prepare_datamart = BashOperator(
        task_id="prepare_datamart",
        bash_command="python main.py --stage all",
        cwd="/app",
    )
    build_model_bank = BashOperator(
        task_id="build_model_bank",
        bash_command="python model_train.py --save-model",
        cwd="/app",
    )
    monthly_inference = BashOperator(
        task_id="monthly_inference",
        bash_command='python model_inference.py --snapshot-date "{{ ds }}"',
        cwd="/app",
    )
    monthly_monitoring = BashOperator(
        task_id="monthly_monitoring",
        bash_command=(
            'python model_monitoring.py --snapshot-date "{{ ds }}" '
            '--as-of-date "{{ logical_date.add(months=6).format(\'YYYY-MM-DD\') }}"'
        ),
        cwd="/app",
    )

    prepare_datamart >> build_model_bank >> monthly_inference >> monthly_monitoring
