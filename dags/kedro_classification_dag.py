from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="kedro_classification_pipeline",
    description="Ejecuta solo el pipeline de clasificación",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
) as dag:
    run_classification = BashOperator(
        task_id="run_classification",
        bash_command="cd /opt/airflow && kedro run --pipeline=classification",
    )
