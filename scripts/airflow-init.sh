#!/usr/bin/env bash
export AIRFLOW_HOME=/opt/airflow

echo "Inicializando Base de Datos de Airflow..."
airflow db init

echo "Creando usuario administrador..."
airflow users create \
  --username airflow \
  --password airflow \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

echo "Inicialización finalizada."
