# src/proyecto_ml/pipelines/classification/nodes.py
"""Nodos de clasificación binaria (e.g., Diabetes 0/1) con GridSearchCV y ROC/AUC."""
from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy


# ======================================================================
# Helpers
# ======================================================================
def _to_binary_series(y: pd.Series) -> pd.Series:
    """
    Convierte un target potencialmente multicategoría a binario 0/1.

    Regla específica para diabetes: 'No diabetes' -> 0, cualquier otro tipo -> 1.
    Si ya es binario (0/1 o 2 etiquetas), intenta mapear de forma segura.
    """
    uniques = pd.Series(pd.unique(y.dropna())).astype(str).str.strip()
    n_classes = len(uniques)

    # Caso ya binario numérico
    if set(pd.unique(y)) <= {0, 1}:
        return y.astype(int)

    # Caso binario con etiquetas textuales: intentar mapear "negativos" conocidos
    if n_classes == 2:
        y_str = y.astype(str).str.lower().str.strip()
        negatives = {"no", "no diabetes", "negativo", "false", "f", "0"}
        mapped = y_str.map(lambda v: 0 if v in negatives else 1)
        # Si el mapeo genera NaN, forzar una de las clases como 0 y la otra 1
        if mapped.isna().any():
            cls = sorted(pd.unique(y_str))
            mapping = {cls[0]: 0, cls[1]: 1}
            mapped = y_str.map(mapping)
        return mapped.astype(int)

    # Caso multicategoría típico de diabetes
    y_str = y.astype(str).str.strip()
    if "No diabetes" in set(uniques):
        return (y_str != "No diabetes").astype(int)

    # No sabemos binarizar de manera segura
    raise ValueError(
        f"El target '{y.name}' tiene {n_classes} clases {list(uniques)}. "
        "Este pipeline es binario. Crea una columna binaria o ajusta _to_binary_series()."
    )


# ======================================================================
# Split
# ======================================================================
def split_supervised(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa X/y y hace train/test split estratificado para clasificación."""
    assert target_col in df.columns, (
        f"Target '{target_col}' no está en columnas: {list(df.columns)[:12]}..."
    )
    y_raw = df[target_col]
    y = _to_binary_series(
        y_raw
    )  # binariza si hace falta (No diabetes=0, resto=1)
    X = df.drop(columns=[target_col])

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ======================================================================
# Preprocesamiento
# ======================================================================
def make_preprocessor() -> ColumnTransformer:
    """Escala numéricas y one-hot de categóricas con salida densa (para SVC/KNN)."""
    num_selector = make_column_selector(dtype_include=np.number)
    cat_selector = make_column_selector(dtype_exclude=np.number)

    # Compatible con versiones viejas/nuevas de scikit-learn
    try:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )  # >=1.2
    except TypeError:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )  # <=1.1

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_selector),
            ("cat", ohe, cat_selector),
        ],
        remainder="drop",
    )


# ======================================================================
# Modelos + grids
# ======================================================================
SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}


def build_model_space(random_state: int) -> Dict[str, Tuple[SkPipeline, Dict[str, list]]]:
    """
    Define el espacio de modelos (5 clasificadores) y sus grids de hiperparámetros.

    Todos comparten el mismo preprocesador (numéricas escaladas + categóricas one-hot),
    lo que simplifica el pipeline y evita fugas de información.
    """
    space: Dict[str, Tuple[SkPipeline, Dict[str, list]]] = {}
    pre = make_preprocessor()

    # 1) LogisticRegression
    pipe_lr = SkPipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    grid_lr = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }
    space["logreg"] = (pipe_lr, grid_lr)

    # 2) RandomForest
    pipe_rf = SkPipeline(
        [
            ("pre", pre),
            (
                "clf",
                RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    grid_rf = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 8, 16],
    }
    space["rf"] = (pipe_rf, grid_rf)

    # 3) GradientBoosting
    pipe_gb = SkPipeline(
        [
            ("pre", pre),
            ("clf", GradientBoostingClassifier(random_state=random_state)),
        ]
    )
    grid_gb = {
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    }
    space["gb"] = (pipe_gb, grid_gb)

    # 4) SVC (probability=True para poder calcular ROC/AUC)
    pipe_svc = SkPipeline(
        [
            ("pre", pre),
            ("clf", SVC(probability=True, random_state=random_state)),
        ]
    )
    grid_svc = {
        "clf__C": [0.5, 1.0, 5.0],
        "clf__kernel": ["rbf", "linear"],
    }
    space["svc"] = (pipe_svc, grid_svc)

    # 5) KNN
    pipe_knn = SkPipeline(
        [
            ("pre", pre),
            ("clf", KNeighborsClassifier()),
        ]
    )
    grid_knn = {
        "clf__n_neighbors": [3, 5, 11],
    }
    space["knn"] = (pipe_knn, grid_knn)

    return space


# ======================================================================
# Entrenamiento y selección
# ======================================================================
def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> Tuple[Dict[str, Any], str]:
    """
    Entrena los 5 modelos con GridSearchCV y devuelve:
    - dict con resultados de CV por modelo (AUC, F1, etc.).
    - nombre del mejor modelo según AUC-ROC promedio en CV.

    Esto alimenta tanto:
    - la selección del mejor modelo,
    - como los gráficos/tablas del reporte (classification_model_compare).
    """
    models = build_model_space(random_state)
    best_name, best_auc = None, -np.inf
    results: Dict[str, Any] = {}

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv,
            scoring=SCORING,
            refit="roc_auc",
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
            error_score=np.nan,
        )
        gs.fit(X_train, y_train)

        # Índice del mejor set de hiperparámetros
        best_idx = gs.best_index_

        # Medias de CV para cada métrica en el mejor punto
        mean_auc = gs.cv_results_["mean_test_roc_auc"][best_idx]
        mean_f1 = gs.cv_results_["mean_test_f1"][best_idx]
        mean_acc = gs.cv_results_["mean_test_accuracy"][best_idx]
        mean_pre = gs.cv_results_["mean_test_precision"][best_idx]
        mean_rec = gs.cv_results_["mean_test_recall"][best_idx]

        results[name] = {
            "best_params": gs.best_params_,
            "cv_auc_mean": float(mean_auc),
            "cv_f1_mean": float(mean_f1),
            "cv_accuracy_mean": float(mean_acc),
            "cv_precision_mean": float(mean_pre),
            "cv_recall_mean": float(mean_rec),
        }

        if np.isfinite(mean_auc) and mean_auc > best_auc:
            best_auc = mean_auc
            best_name = name

    if best_name is None or not np.isfinite(best_auc):
        raise ValueError(
            "Todos los modelos fallaron al puntuar (AUC no definido). "
            "Revisa que el target sea binario tras el split y que no haya valores nulos."
        )

    results["_best_name"] = best_name
    results["_best_cv_auc"] = float(best_auc)
    return results, best_name


# ======================================================================
# Refit del mejor modelo
# ======================================================================
def refit_best(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    cv_results: Dict[str, Any],
    best_name: str,
):
    """
    Reconstruye el pipeline ganador con sus mejores hiperparámetros
    y lo entrena en todo el set de entrenamiento.
    """
    space = build_model_space(random_state)
    if best_name not in space:
        raise ValueError(f"Modelo '{best_name}' no existe en el espacio de modelos.")
    pipe, _ = space[best_name]
    best_params = cv_results[best_name]["best_params"]
    model = deepcopy(pipe).set_params(**best_params)
    model.fit(X_train, y_train)
    return model


# ======================================================================
# Evaluación en test
# ======================================================================
def evaluate_on_test(
    best_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], plt.Figure]:
    """
    Evalúa el mejor modelo en el set de test y devuelve:
    - dict con métricas (accuracy, precision, recall, f1, auc_roc, confusion_matrix)
    - figura con la curva ROC
    """
    # Probabilidades o scores
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        y_proba = best_model.decision_function(X_test)
        # Escalamos a [0,1] si viene como márgenes
        if np.issubdtype(np.asarray(y_proba).dtype, np.number):
            y_proba = (y_proba - np.min(y_proba)) / (
                np.max(y_proba) - np.min(y_proba) + 1e-9
            )
    else:
        # Fallback: usamos predicción binaria como "score"
        y_proba = best_model.predict(X_test)

    # Predicción binaria por umbral 0.5 si y_proba es 1D
    if np.ndim(y_proba) == 1:
        y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    else:
        y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics["auc_roc"] = float("nan")

    # Matriz de confusión (como lista para poder guardarla en JSON)
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    # Curva ROC
    fig, ax = plt.subplots()
    try:
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("ROC - Test")
    except Exception:
        ax.text(0.5, 0.5, "ROC no disponible", ha="center")

    return metrics, fig


# ======================================================================
# Comparación de modelos (para reporting)
# ======================================================================
def compare_models_from_cv(
    cv_results: Dict[str, Any],
) -> Tuple[pd.DataFrame, plt.Figure, plt.Figure]:
    """
    Construye:
      - una tabla comparativa de modelos (AUC, F1, accuracy, precision, recall),
      - una figura de barras para AUC,
      - una figura de barras para F1.

    Se alimenta directamente de classification_cv_results.json.
    """
    rows = []
    for model_name, info in cv_results.items():
        if model_name.startswith("_"):
            continue

        rows.append(
            {
                "model": str(model_name),
                "auc_mean": float(info.get("cv_auc_mean", np.nan)),
                "f1_mean": float(info.get("cv_f1_mean", np.nan)),
                "accuracy_mean": float(info.get("cv_accuracy_mean", np.nan)),
                "precision_mean": float(info.get("cv_precision_mean", np.nan)),
                "recall_mean": float(info.get("cv_recall_mean", np.nan)),
            }
        )

    if not rows:
        df = pd.DataFrame([{"model": "N/A", "auc_mean": np.nan}])
    else:
        df = pd.DataFrame(rows).sort_values(
            by="auc_mean", ascending=False, na_position="last"
        )

    # ---------- AUC ----------
    fig_auc, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df["auc_mean"])
    ax.set_title("Comparativa modelos - AUC (CV)")
    ax.set_ylabel("AUC (mean CV)")
    ax.set_xlabel("Modelo")

    auc_vals = df["auc_mean"].dropna().astype(float)
    if len(auc_vals) > 0:
        ymin = max(0.0, auc_vals.min() - 0.05)
        ymax = min(1.0, auc_vals.max() + 0.02)
        if ymin < ymax:
            ax.set_ylim(ymin, ymax)

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    for rect, val in zip(ax.patches, df["auc_mean"].tolist()):
        if not pd.isna(val):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

    # ---------- F1 ----------
    fig_f1, ax2 = plt.subplots(figsize=(8, 5))
    if df["f1_mean"].notna().any():
        ax2.bar(df["model"], df["f1_mean"])
        ax2.set_title("Comparativa modelos - F1 (CV)")
        ax2.set_ylabel("F1 (mean CV)")
        ax2.set_xlabel("Modelo")

        f1_vals = df["f1_mean"].dropna().astype(float)
        if len(f1_vals) > 0:
            ymin = max(0.0, f1_vals.min() - 0.05)
            ymax = min(1.0, f1_vals.max() + 0.02)
            if ymin < ymax:
                ax2.set_ylim(ymin, ymax)

        for lbl in ax2.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")

        for rect, val in zip(ax2.patches, df["f1_mean"].tolist()):
            if not pd.isna(val):
                ax2.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                )
    else:
        ax2.text(
            0.5,
            0.5,
            "No hay F1 en classification_cv_results",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax2.set_axis_off()

    return df, fig_auc, fig_f1
