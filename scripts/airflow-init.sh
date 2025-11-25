#!/usr/bin/env bash
set -Eeuo pipefail
echo "===> Migrando DB..."
airflow db migrate
echo "===> Creando usuario admin..."
airflow users create \
  --username "${_AIRFLOW_WWW_USER_USERNAME}" \
  --firstname "Rox" \
  --lastname "Ana" \
  --role "Admin" \
  --email "admin@example.com" \
  --password "${_AIRFLOW_WWW_USER_PASSWORD}" || true
echo "===> Listo."
