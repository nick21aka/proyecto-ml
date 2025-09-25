"""Pipeline de la Fase 3 (Data Preparation).

Crea tres nodos `prep`, uno por dataset crudo, y guarda:
- *_clean  → capa 03_primary
- *_features → capa 04_feature
"""
from kedro.pipeline import Pipeline, node
from .nodes import prep


def create_pipeline(**kwargs) -> Pipeline:
    """Construye el pipeline de preparación de datos.

    Returns:
        Pipeline: pipeline con nodos para diabetes, cardio y cardiabetes.
    """
    return Pipeline(
        [
            node(
                func=prep,
                inputs="diabetes_raw",
                outputs=["diabetes_clean", "diabetes_features"],
                name="prep_diabetes",
            ),
            node(
                func=prep,
                inputs="cardio_raw",
                outputs=["cardio_clean", "cardio_features"],
                name="prep_cardio",
            ),
            node(
                func=prep,
                inputs="cardiabetes_raw",
                outputs=["cardiabetes_clean", "cardiabetes_features"],
                name="prep_cardiabetes",
            ),
        ]
    )
