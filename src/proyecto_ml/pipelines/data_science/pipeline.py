"""Pipeline de ‘data_science’: split train/test para 3 datasets."""
from kedro.pipeline import Pipeline, node
from .nodes import split_dataset


def create_pipeline(**kwargs) -> Pipeline:
    """Construye el pipeline de split para diabetes, cardio y cardiabetes."""
    return Pipeline([
        node(
            func=split_dataset,
            inputs=["diabetes_features", "params:targets.diabetes", "params:split.test_size", "params:split.random_state"],
            outputs=["diabetes_X_train", "diabetes_X_test", "diabetes_y_train", "diabetes_y_test"],
            name="split_diabetes",
        ),
        node(
            func=split_dataset,
            inputs=["cardio_features", "params:targets.cardio", "params:split.test_size", "params:split.random_state"],
            outputs=["cardio_X_train", "cardio_X_test", "cardio_y_train", "cardio_y_test"],
            name="split_cardio",
        ),
        node(
            func=split_dataset,
            inputs=["cardiabetes_features", "params:targets.cardiabetes", "params:split.test_size", "params:split.random_state"],
            outputs=["cardiabetes_X_train", "cardiabetes_X_test", "cardiabetes_y_train", "cardiabetes_y_test"],
            name="split_cardiabetes",
        ),
    ])
