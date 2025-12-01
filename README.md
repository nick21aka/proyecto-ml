# Proyecto ML — CRISP-DM (Kedro + Airflow + DVC + Docker)

Prototipo reproducible de análisis, preparación y modelamiento de datos clínicos utilizando **Kedro**, **Airflow**, **DVC** y **Docker**, siguiendo la metodología **CRISP–DM** (Fases 1–3) y ampliado con técnicas de **Aprendizaje No Supervisado**.

---

## 🧠 Descripción General

Este proyecto implementa un pipeline completo de *data engineering* y *machine learning* para datasets clínicos (diabetes y riesgo cardiovascular).  
Incluye:

- Limpieza y preparación (CRISP-DM Fase 2–3)
- Feature engineering automatizado (Kedro)
- Orquestación de pipelines (Airflow)
- Versionado de datos y modelos (DVC)
- Visualizaciones interactivas (Plotly / Notebooks)
- Técnicas avanzadas de aprendizaje no supervisado:
  - Clustering
  - Reducción de dimensionalidad
  - Detección de anomalías (opcional)

---

# 🏗️ Arquitectura del Proyecto Final

## 🔹 Framework: **Kedro**
- **Pipeline integrado:** `unsupervised_learning/`
- **Catálogo actualizado:** datasets versionados y declarados en `catalog.yml`
- **Parámetros configurables:** mediante `parameters.yml` (KMeans, DBSCAN, PCA, etc.)

## 🔹 Orquestación: **Apache Airflow**
- **DAG principal:** `data_engineering → supervised → unsupervised`
- **Tasks independientes:** ejecución modular por algoritmo
- **Control de dependencias:** upstream / downstream para reproducibilidad

## 🔹 Versionado: **DVC**
- Versionado de:
  - Features de clustering  
  - Modelos de reducción dimensional  
  - Métricas de experimentos (silhouette, DBI, CH, inertia, etc.)

## 🔹 Contenedores: **Docker**
- **Dockerfile** actualizado
- **docker-compose.airflow.yml** completo
- Servicios incluidos:
  - Airflow webserver
  - Scheduler
  - Init
  - Worker
- **Volúmenes configurados** para logs, metadatos, DVC y pipelines

---

# 🤖 Técnicas de Aprendizaje No Supervisado

## 1) **Clustering (OBLIGATORIO)**  
Se implementan **al menos 3 algoritmos**, comparando desempeño:

- **K-Means**
- **DBSCAN**
- **Hierarchical Clustering (Aglomerativo)**
- (Opcional) Gaussian Mixture Models
- (Opcional) OPTICS

### **Métricas obligatorias:**
- Silhouette Score  
- Davies–Bouldin Index  
- Calinski–Harabasz Index  
- Elbow Method  
- Dendrogramas (para clustering jerárquico)

---

## 2) **Reducción de Dimensionalidad (OBLIGATORIO)**  
Implementación de al menos 2 métodos:

- **PCA:** varianza explicada, loadings, biplots 2D/3D
- **t-SNE:** proyección no lineal para alta dimensión
- **UMAP:** alternativa moderna a t-SNE
- **Truncated SVD** (para datos sparse)

---

## 3) **Detección de Anomalías (OPCIONAL)**
- Isolation Forest  
- Local Outlier Factor (LOF)  
- One-Class SVM  

---

# 📦 Requisitos

- Python **3.10+** (probado con 3.13)
- Windows 10/11
- Pip y venv
- Docker Desktop (para Airflow)
- Git + DVC (opcional)

---

# ⚙️ Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
