from __future__ import annotations
from kedro.pipeline import Pipeline, node
from . import nodes


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            # 1) Split supervisado
            node(
                func=nodes.split_supervised,
                inputs=[
                    "regression_features",
                    "params:regression.target",
                    "params:split_reg.test_size",
                    "params:split_reg.random_state",
                ],
                outputs=[
                    "reg_X_train",
                    "reg_X_test",
                    "reg_y_train",
                    "reg_y_test",
                ],
                name="reg_split",
            ),

            # 2) Entrenamiento + selección del mejor modelo
            node(
                func=nodes.train_and_select,
                inputs=[
                    "reg_X_train",
                    "reg_y_train",
                    "params:regression.cv_folds",
                    "params:split_reg.random_state",
                ],
                outputs=[
                    "regression_cv_results",
                    "regression_best_name",
                ],
                name="reg_train_select",
            ),

            # 3) Comparativa de modelos: tabla + gráficos R² / RMSE
            node(
                func=nodes.compare_models_from_cv,
                inputs="regression_cv_results",
                outputs=[
                    "regression_model_compare",
                    "regression_r2_bar",
                    "regression_rmse_bar",
                ],
                name="reg_compare_models",
            ),

            # 4) Refit del mejor modelo
            node(
                func=nodes.refit_best,
                inputs=[
                    "reg_X_train",
                    "reg_y_train",
                    "params:split_reg.random_state",
                    "regression_cv_results",
                    "regression_best_name",
                ],
                outputs="best_regressor",
                name="reg_refit_best",
            ),

            # 5) Evaluación en test + CSV + figuras
            node(
                func=nodes.evaluate_on_test,
                inputs=[
                    "best_regressor",
                    "reg_X_test",
                    "reg_y_test",
                ],
                outputs=[
                    "regression_test_metrics",
                    "regression_pred_vs_true",
                    "regression_pred_vs_true_plot",
                    "regression_resid_plot",
                ],
                name="reg_eval_test",
            ),
        ]
    )
