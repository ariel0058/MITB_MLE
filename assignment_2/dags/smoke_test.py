"""Minimal manual DAG to verify that Airflow loads and executes project DAGs."""

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="assignment2_smoke_test",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["assignment2", "smoke-test"],
)
def smoke_test():
    @task
    def confirm_airflow():
        print("Assignment 2 Airflow smoke test passed")

    confirm_airflow()


smoke_test()
