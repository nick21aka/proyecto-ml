from __future__ import annotations

from kedro.pipeline import Pipeline

from proyecto_ml.pipelines.data_engineering import pipeline as de_pipeline
from proyecto_ml.pipelines.classification import pipeline as classification_pipeline
from proyecto_ml.pipelines.regression import pipeline as regression_pipeline
from proyecto_ml.pipelines.clustering import pipeline as clustering_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Registra los pipelines disponibles en el proyecto."""

    # 🔹 Pipelines individuales
    data_engineering = de_pipeline.create_pipeline()
    classification = classification_pipeline.create_pipeline()
    regression = regression_pipeline.create_pipeline()
    unsupervised = clustering_pipeline.create_pipeline()

    # 🔹 Pipeline supervisado (clasificación + regresión)
    supervised = classification + regression

    # 🔹 Pipeline completo (para el DAG proyecto_ml_full_pipeline)
    full_pipeline = data_engineering + supervised + unsupervised

    return {
        "data_engineering": data_engineering,
        "classification": classification,
        "regression": regression,
        "supervised_learning": supervised,
        "unsupervised_learning": unsupervised,
        "__default__": full_pipeline,
    }
