"""
Nodos de regresión supervisada (e.g., Colesterol_Total) con GridSearchCV.
Incluye:
- 5 modelos
- CV k-fold
- Métricas R², RMSE, MAE
- Real vs Predicho
- Residuales
- Tabla comparativa + barplots
"""

from __future__ import annotations
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline as SkPipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from copy import deepcopy


# ======================================================================
# 1. SPLIT
# ======================================================================
def split_supervised(df, target, test_size, random_state):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Convertimos Series → DataFrame para que Kedro pueda guardarlos
    y_train = pd.DataFrame(y_train, columns=[target])
    y_test = pd.DataFrame(y_test, columns=[target])

    return X_train, X_test, y_train, y_test


# ======================================================================
# 2. PREPROCESAMIENTO
# ======================================================================
def make_preprocessor() -> ColumnTransformer:
    """ColumnTransformer con StandardScaler para numéricas y OHE para categóricas."""

    num_selector = make_column_selector(dtype_include=np.number)
    cat_selector = make_column_selector(dtype_exclude=np.number)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Compatibilidad con versiones anteriores de sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_selector),
            ("cat", ohe, cat_selector),
        ],
        remainder="drop",
    )


# ======================================================================
# 3. DEFINICIÓN DE MODELOS + GRIDS
# ======================================================================
SCORING = {
    "r2": "r2",
    "neg_mse": "neg_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
}


def build_model_space(random_state: int) -> Dict[str, Tuple[SkPipeline, Dict[str, list]]]:
    """Define los 5 modelos de regresión + grids de hiperparámetros."""

    pre = make_preprocessor()
    space: Dict[str, Tuple[SkPipeline, Dict[str, list]]] = {}

    # 1) Linear Regression (sin grid)
    pipe_lr = SkPipeline([("pre", pre), ("reg", LinearRegression())])
    grid_lr: Dict[str, list] = {}
    space["linreg"] = (pipe_lr, grid_lr)

    # 2) RandomForestRegressor
    pipe_rf = SkPipeline(
        [
            ("pre", pre),
            # n_jobs=1 para que no consuma demasiados recursos en el contenedor
            ("reg", RandomForestRegressor(random_state=random_state, n_jobs=1)),
        ]
    )
    grid_rf = {
        "reg__n_estimators": [200, 400],
        "reg__max_depth": [None, 8, 16],
    }
    space["rf"] = (pipe_rf, grid_rf)

    # 3) GradientBoostingRegressor
    pipe_gb = SkPipeline(
        [("pre", pre), ("reg", GradientBoostingRegressor(random_state=random_state))]
    )
    grid_gb = {
        "reg__n_estimators": [200, 400],
        "reg__learning_rate": [0.05, 0.1],
        "reg__max_depth": [2, 3],
    }
    space["gb"] = (pipe_gb, grid_gb)

    # 4) KNeighborsRegressor
    pipe_knn = SkPipeline([("pre", pre), ("reg", KNeighborsRegressor())])
    grid_knn = {
        "reg__n_neighbors": [3, 5, 11],
        "reg__weights": ["uniform", "distance"],
    }
    space["knn"] = (pipe_knn, grid_knn)

    # 5) SVR
    pipe_svr = SkPipeline([("pre", pre), ("reg", SVR())])
    grid_svr = {
        "reg__C": [0.5, 1.0, 5.0],
        "reg__epsilon": [0.1, 0.2],
        "reg__kernel": ["rbf", "linear"],
    }
    space["svr"] = (pipe_svr, grid_svr)

    return space


# ======================================================================
# 4. ENTRENAR Y SELECCIONAR MODELO
# ======================================================================
def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,  # ahora puede venir como DataFrame (1 col)
    cv_folds: int,
    random_state: int,
) -> Tuple[Dict[str, Any], str]:

    models = build_model_space(random_state)
    cv_results: Dict[str, Any] = {}
    best_name: str | None = None
    best_r2 = -np.inf

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # y_train como array 1D por seguridad
    y_arr = np.asarray(y_train).ravel()

    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv,
            scoring=SCORING,
            refit="r2",
            n_jobs=1,          # ⚠️ IMPORTANTE: 1 worker dentro del contenedor
            verbose=0,
            return_train_score=True,
        )
        gs.fit(X_train, y_arr)

        idx = gs.best_index_

        mean_r2 = gs.cv_results_["mean_test_r2"][idx]
        mean_neg_mse = gs.cv_results_["mean_test_neg_mse"][idx]
        mean_neg_mae = gs.cv_results_["mean_test_neg_mae"][idx]

        mean_rmse = float(np.sqrt(-mean_neg_mse))
        mean_mae = float(-mean_neg_mae)

        cv_results[name] = {
            "best_params": gs.best_params_,
            "cv_r2_mean": float(mean_r2),
            "cv_rmse_mean": mean_rmse,
            "cv_mae_mean": mean_mae,
        }

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_name = name

    cv_results["_best_name"] = best_name
    cv_results["_best_cv_r2"] = float(best_r2)

    return cv_results, best_name  # type: ignore[return-value]


# ======================================================================
# 5. REFIT
# ======================================================================
def refit_best(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,  # también DataFrame 1 col
    random_state: int,
    cv_results: Dict[str, Any],
    best_name: str,
):
    space = build_model_space(random_state)

    if best_name not in space:
        raise ValueError(f"Modelo '{best_name}' no existe en el espacio.")

    pipe, _ = space[best_name]
    best_params = cv_results[best_name]["best_params"]

    model = deepcopy(pipe).set_params(**best_params)

    y_arr = np.asarray(y_train).ravel()
    model.fit(X_train, y_arr)

    return model


# ======================================================================
# 6. EVALUACIÓN EN TEST
# ======================================================================
def evaluate_on_test(best_model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    # Asegurar 1D
    y_true = np.asarray(y_test).ravel()
    y_pred = best_model.predict(X_test)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    # CSV: real vs predicho
    df_pred = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    # Figura: Real vs Predicho
    fig_pred, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4)
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        "r--",
        label="Línea ideal",
    )
    ax.set_title("Real vs Predicho (Test)")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    ax.legend()
    fig_pred.tight_layout()

    # Figura: residuales
    resid = y_true - y_pred
    fig_resid, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(y_pred, resid, alpha=0.4)
    ax2.axhline(0.0, color="r", linestyle="--", label="Residuo = 0")
    ax2.set_title("Residuales vs Predicciones")
    ax2.set_xlabel("Predicción")
    ax2.set_ylabel("Residuo (y_real - y_pred)")
    ax2.legend()
    fig_resid.tight_layout()

    return metrics, df_pred, fig_pred, fig_resid


# ======================================================================
# 7. COMPARATIVA (R² y RMSE)
# ======================================================================
def compare_models_from_cv(
    cv_results: Dict[str, Any],
) -> Tuple[pd.DataFrame, plt.Figure, plt.Figure]:

    rows = []
    for name, info in cv_results.items():
        if name.startswith("_"):
            continue
        rows.append(
            {
                "model": name,
                "r2_mean": info["cv_r2_mean"],
                "rmse_mean": info["cv_rmse_mean"],
                "mae_mean": info["cv_mae_mean"],
            }
        )

    df = pd.DataFrame(rows).sort_values(by="r2_mean", ascending=False)

    # --- Gráfico R² ---
    fig_r2, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df["r2_mean"])
    ax.set_title("Comparación de Modelos - R² (CV)")
    ax.set_ylabel("R² (CV)")
    ax.set_xlabel("Modelo")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    # --- Gráfico RMSE ---
    fig_rmse, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(df["model"], df["rmse_mean"])
    ax2.set_title("Comparación de Modelos - RMSE (CV)")
    ax2.set_ylabel("RMSE ↓ (mejor más bajo)")
    ax2.set_xlabel("Modelo")
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    return df, fig_r2, fig_rmse
