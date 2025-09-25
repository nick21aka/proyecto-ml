from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_not_empty, basic_clean

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(validate_not_empty, dict(df="diabetes_raw", name="params:targets.diabetes"), "diabetes_valid"),
        node(basic_clean, "diabetes_valid", "diabetes_clean"),

        node(validate_not_empty, dict(df="cardio_raw", name="params:targets.cardio"), "cardio_valid"),
        node(basic_clean, "cardio_valid", "cardio_clean"),

        node(validate_not_empty, dict(df="cardiabetes_raw", name="params:targets.cardiabetes"), "cardiabetes_valid"),
        node(basic_clean, "cardiabetes_valid", "cardiabetes_clean"),
    ])
