# src/proyecto_ml/pipeline_registry.py
from __future__ import annotations
from kedro.pipeline import Pipeline

# Pipelines principales
from proyecto_ml.pipelines.dataprep import create_pipeline as create_dataprep_pipeline
from proyecto_ml.pipelines.classification import create_pipeline as create_classification_pipeline

# Import seguro de regression
try:
    from proyecto_ml.pipelines.regression import create_pipeline as create_regression_pipeline
    _HAS_REGRESSION = True
except Exception as e:
    print(f"[pipeline_registry] Aviso: no se pudo importar regression: {e}")
    _HAS_REGRESSION = False


def register_pipelines() -> dict[str, Pipeline]:
    """Registra todos los pipelines disponibles en el proyecto."""

    dp = create_dataprep_pipeline()
    cls = create_classification_pipeline()

    pipelines: dict[str, Pipeline] = {
        "dataprep": dp,
        "classification": cls,
        "__default__": cls,
    }

    if _HAS_REGRESSION:
        try:
            reg = create_regression_pipeline()
        except Exception as e:
            print(f"[pipeline_registry] Error al construir pipeline regression: {e}")
        else:
            pipelines["regression"] = reg

    return pipelines
