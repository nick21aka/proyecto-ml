from kedro.pipeline import Pipeline, node, pipeline
from .nodes import report_for_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=report_for_dataset,
            inputs=dict(
                df="diabetes_raw",
                target_col="params:targets.diabetes",
                name="params:reporting_names.diabetes",
            ),
            outputs=["diabetes_target_plot", "diabetes_hists_plot"],
            name="report_diabetes_raw",
        ),
        node(
            func=report_for_dataset,
            inputs=dict(
                df="cardio_raw",
                target_col="params:targets.cardio",
                name="params:reporting_names.cardio",
            ),
            outputs=["cardio_target_plot", "cardio_hists_plot"],
            name="report_cardio_raw",
        ),
        node(
            func=report_for_dataset,
            inputs=dict(
                df="cardiabetes_raw",
                target_col="params:targets.cardiabetes",
                name="params:reporting_names.cardiabetes",
            ),
            outputs=["cardiabetes_target_plot", "cardiabetes_hists_plot"],
            name="report_cardiabetes_raw",
        ),
    ])
