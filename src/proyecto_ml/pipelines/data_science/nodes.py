"""Nodos de la Fase 3½ (Data Science “prep”): split train/test.

Este módulo NO entrena modelos (tu rúbrica llega hasta Fase 3).
Sólo toma las tablas de *features* y las divide en conjuntos
de entrenamiento y prueba, manteniendo el target aparte.
"""
from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide un DataFrame en X_train, X_test, y_train, y_test.

    Si el target es binario, realiza *stratify* para mantener la proporción.

    Args:
        df: DataFrame con *features* + columna target.
        target_col: Nombre de la columna objetivo.
        test_size: Proporción del set de prueba.
        random_state: Semilla de aleatoriedad.

    Returns:
        (X_train, X_test, y_train_df, y_test_df) donde y_* son DataFrames
        de una sola columna (target_col) para facilitar el guardado a Parquet.
    """
    assert target_col in df.columns, f"Target '{target_col}' no está en columnas: {list(df.columns)[:10]}..."
    y = df[target_col]
    X = df.drop(columns=[target_col])

    stratify = y if y.dropna().nunique() in (2, 3) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Devolver y_* como DataFrame de una columna para guardado estable en Parquet
    y_tr = y_tr.to_frame(name=target_col)
    y_te = y_te.to_frame(name=target_col)
    return X_tr, X_te, y_tr, y_te
