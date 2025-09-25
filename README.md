# Proyecto ML — CRISP-DM (Kedro)

## Descripción
Prototipo reproducible para análisis y preparación de datos clínicos (diabetes y cardiovascular), siguiendo **CRISP–DM** (Fases 1–3).

## Requisitos
- Python 3.10+ (probado con 3.13)
- Windows 10/11
- Pip y venv

## Instalación
```bash
# Crear y activar entorno virtual (Windows PowerShell)
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

#Estructura de datos

data/01_raw/: entrada (CSV crudos)

db_diabetes.csv, db_cardio.csv, db_cardiabetes.csv

data/03_primary/: salida de limpieza (Parquet) — generado por pipeline

data/04_feature/: salida de features (Parquet) — generado por pipeline

data/08_reporting/: reportes EDA (CSV/PNG) — generado desde notebooks

#Ejecutar pipelines

# Preparación de datos para los 3 datasets
kedro run --pipeline=dataprep

# (Opcional) Visualizar el grafo del proyecto
kedro viz

#Notebooks

notebooks/01_business_understanding.ipynb — Fase 1

notebooks/02_data_understanding.ipynb — Fase 2 (EDA + reportes)

notebooks/03_data_preparation.ipynb — Fase 3 (limpieza + features)


# Parámetros clave

Costos de error (criterio): FN = 10, FP = 2

Targets por dataset (preferidos / proxies):

diabetes_raw: Diabetes/Outcome; si no existe → Diabetes_proxy por HbA1c ≥ 6.5

cardio_raw: Riesgo_Alto (derivado de Riesgo_Cardiovascular), o cardio_proxy (PA/colesterol/glucosa)

cardiabetes_raw: Diabetes (0/1) o Riesgo_Alto si aplica

# Reproducibilidad

# Activar entorno
venv\Scripts\activate

# Ejecutar pipeline de preparación
kedro run --pipeline=dataprep

# (Opcional) Generar reportes EDA desde notebooks
# - diccionario de datos y resumen EDA a CSV
# - histogramas y correlación a PNG

# Artefactos generados (Fase 2–3)

data/03_primary/*_clean.parquet

data/04_feature/*_features.parquet

data/08_reporting/:

*_data_dictionary.csv (diccionario de datos)

*_eda_summary.csv (resumen filas/columnas/nulos/duplicados)

*_hists.png (histogramas)

*_corr.png (mapa de correlación)

# Estructura del proyecto (resumen)

proyecto-ml/
├─ conf/
│  └─ base/
│     ├─ catalog.yml
│     ├─ parameters.yml
│     └─ parameters_*.yml
├─ data/
│  ├─ 01_raw/
│  ├─ 03_primary/
│  ├─ 04_feature/
│  └─ 08_reporting/
├─ notebooks/
│  ├─ 01_business_understanding.ipynb
│  ├─ 02_data_understanding.ipynb
│  └─ 03_data_preparation.ipynb
├─ src/
│  └─ proyecto_ml/
│     ├─ pipeline_registry.py
│     └─ pipelines/
│        └─ dataprep/
│           ├─ __init__.py
│           ├─ nodes.py
│           └─ pipeline.py
├─ pyproject.toml
├─ requirements.txt
└─ README.md

# Git (opcional, para la entrega)

git init
git add .
git commit -m "Kedro proyecto — Fases 1-3 (EDA + preparación + pipeline dataprep)"
# git remote add origin https://github.com/tu-usuario/proyecto-ml.git
# git branch -M main
# git push -u origin main

#Problemas comunes (Troubleshooting)

Kedro no encuentra pyproject.toml: abre Jupyter en la raíz o os.chdir("..") desde notebooks/.

Guardar Parquet falla: instala motor Parquet

pip install pyarrow


Catálogo no encuentra datasets: revisa conf/base/catalog.yml (los nombres deben coincidir exactos).

catalog.list() no existe: usa name in catalog o sorted(list(catalog)).

## Guía rápida (Windows)

```bash
# 1) Crear entorno e instalar
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2) Colocar los CSV en data/01_raw/
#   db_diabetes.csv, db_cardio.csv, db_cardiabetes.csv

# 3) Ejecutar la preparación
kedro run --pipeline=dataprep

# 4) (Opcional) Abrir Jupyter
pip install jupyterlab
jupyter lab


#Licencia

Uso académico.