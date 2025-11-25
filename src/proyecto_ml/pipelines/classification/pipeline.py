# src/proyecto_ml/pipelines/classification/pipeline.py
"""Pipeline de clasificación binaria (diabetes) con GridSearchCV y ROC/AUC."""
from kedro.pipeline import Pipeline, node
from . import nodes


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            # 1) Split supervisado sobre el dataset de features
            node(
                func=nodes.split_supervised,
                inputs=[
                    "diabetes_features",              # dataset de entrada (04_feature)
                    "params:classification.target",   # nombre EXACTO de la columna target
                    "params:split.test_size",
                    "params:split.random_state",
                ],
                outputs=[
                    "cls_X_train",
                    "cls_X_test",
                    "cls_y_train",
                    "cls_y_test",
                ],
                name="cls_split",
            ),

            # 2) Entrenamiento y selección del mejor modelo (por AUC-ROC)
            node(
                func=nodes.train_and_select,
                inputs=[
                    "cls_X_train",
                    "cls_y_train",
                    "params:classification.cv_folds",  # nº de folds de CV
                    "params:split.random_state",
                ],
                outputs=[
                    "classification_cv_results",   # dict con mejores params y AUC por modelo
                    "classification_best_name",    # nombre del mejor modelo
                ],
                name="cls_train_select",
            ),
            
            node(
            func=nodes.compare_models_from_cv,
            inputs="classification_cv_results",
            outputs=[
                "classification_model_compare",    # tabla CSV
                "classification_auc_bar",         # PNG
                "classification_f1_bar",          # PNG
            ],
            name="cls_compare_models",
            ),
        
        
            # 3) Reajuste (refit) del mejor modelo con sus mejores hiperparámetros
            node(
                func=nodes.refit_best,  # Asegúrate que esta función existe en nodes.py
                inputs=[
                    "cls_X_train",
                    "cls_y_train",
                    "params:split.random_state",
                    "classification_cv_results",
                    "classification_best_name",
                ],
                outputs="best_classifier",
                name="cls_refit_best",
            ),

            # 4) Evaluación en test + curva ROC
            node(
                func=nodes.evaluate_on_test,
                inputs=["best_classifier", "cls_X_test", "cls_y_test"],
                outputs=["classification_test_metrics", "classification_roc_fig"],
                name="cls_eval_test",
            ),
        ]
    )

