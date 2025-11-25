# dags/kedro_pipelines_dag.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"owner": "airflow", "depends_on_past": False, "retries": 0}

with DAG(
    dag_id="kedro_both_pipelines",
    description="Demostración de orquestación (clasificación y regresión)",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # pon "0 2 * * *" si te piden cron
    catchup=False,
    default_args=default_args,
) as dag:

    run_classification = BashOperator(
        task_id="run_classification",
        bash_command='echo "Simulando pipeline de clasificación OK"'
    )

    run_regression = BashOperator(
        task_id="run_regression",
        bash_command='echo "Simulando pipeline de regresión OK"'
    )

    run_classification >> run_regression
