# dags/kedro_pipelines_dag.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

# DAG 1 – PIPELINE DE CLASIFICACIÓN
with DAG(
    dag_id="kedro_classification_pipeline",
    description="Pipeline de clasificación con Kedro",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
):
    BashOperator(
        task_id="run_classification",
        bash_command="cd /opt/airflow && kedro run --pipeline=classification",
    )

# DAG 2 – PIPELINE DE REGRESIÓN
with DAG(
    dag_id="kedro_regression_pipeline",
    description="Pipeline de regresión con Kedro",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
):
    BashOperator(
        task_id="run_regression",
        bash_command="cd /opt/airflow && kedro run --pipeline=regression",
    )

# dags/kedro_pipelines_dag.py
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

# Ojo con la ruta: dentro del contenedor debe ser donde está tu repo
KEDRO_PROJECT_DIR = "/opt/airflow/proyecto-ml"

with DAG(
    dag_id="proyecto_ml_full_pipeline",
    default_args=default_args,
    description="Data engineering -> supervised -> unsupervised (clustering) con Kedro",
    schedule_interval=None,      # lo ejecutas a mano
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    data_engineering = BashOperator(
        task_id="data_engineering",
        bash_command=f"cd {KEDRO_PROJECT_DIR} && kedro run --pipeline=data_engineering",
    )

    supervised = BashOperator(
        task_id="supervised",
        bash_command=f"cd {KEDRO_PROJECT_DIR} && kedro run --pipeline=supervised",
    )

    unsupervised = BashOperator(
        task_id="unsupervised_learning",
        bash_command=f"cd {KEDRO_PROJECT_DIR} && kedro run --pipeline=unsupervised_learning",
    )

    # Dependencias: primero data_engineering, luego supervised, luego unsupervised
    data_engineering >> supervised >> unsupervised
