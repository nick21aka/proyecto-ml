from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="kedro_regression_pipeline",
    description="Ejecuta solo el pipeline de regresión",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
) as dag:
    run_regression = BashOperator(
        task_id="run_regression",
        bash_command="cd /opt/airflow && kedro run --pipeline=regression",
    )
