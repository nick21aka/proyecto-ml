"""Registro de pipelines del proyecto.

Expone:
- "__default__": pipeline por defecto.
- "dataprep": preparación (Fase 3).
- "data_science": split train/test (prep DS).
"""
from kedro.pipeline import Pipeline
from proyecto_ml.pipelines.dataprep import pipeline as dataprep
from proyecto_ml.pipelines.data_science import pipeline as data_science


def register_pipelines() -> dict[str, Pipeline]:
    """Registra y devuelve los pipelines disponibles del proyecto."""
    dp = dataprep.create_pipeline()
    ds = data_science.create_pipeline()
    return {
        "__default__": dp + ds,   # si prefieres que 'kedro run' ejecute ambos
        "dataprep": dp,
        "data_science": ds,
    }

from proyecto_ml.pipeline_registry import register_pipelines
pipes = register_pipelines()
print("Kedro OK — pipelines:", list(pipes))
