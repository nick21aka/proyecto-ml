"""Nodos de la Fase 3 (Data Preparation).

Incluye funciones de limpieza básica, tratamiento simple de outliers,
ingeniería de variables ligera y un nodo `prep` que devuelve
dos artefactos por dataset: (clean, features).

Estilo de docstrings: Google.
"""
from typing import Tuple
import numpy as np
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas (snake_case seguro).

    - Quita espacios al inicio/fin.
    - Reemplaza espacios internos por '_'.
    - Elimina caracteres no alfanuméricos (mantiene solo [0-9a-zA-Z_]).

    Args:
        df: DataFrame original.

    Returns:
        DataFrame con nombres de columnas normalizados.
    """
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("[^0-9a-zA-Z_]", "", regex=True)
    )
    return out


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica del dataset.

    Pasos:
      1) Normaliza nombres de columnas.
      2) Elimina duplicados completos.
      3) Para columnas tipo texto: aplica strip() y convierte vacíos a NA.
      4) Intenta convertir texto numérico (p. ej. '1,23') a numérico.

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame limpio.
    """
    before = len(df)
    df = _normalize_columns(df)
    df = df.drop_duplicates().copy()

    obj = df.select_dtypes(include=["object", "string"]).columns
    for c in obj:
        df[c] = df[c].astype("string").str.strip().replace({"": pd.NA})

    # Coerción numérica suave (sin romper strings que no sean números)
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="ignore")
            except Exception:
                pass

    return df


def iqr_clip(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Aplica clipping por IQR a columnas numéricas para reducir outliers.

    Para cada columna numérica:
      low = Q1 - factor*IQR
      high = Q3 + factor*IQR
      Se hace clip al rango [low, high].

    Args:
        df: DataFrame de entrada.
        factor: Multiplicador del IQR (por defecto 3.0).

    Returns:
        DataFrame con columnas numéricas recortadas.
    """
    out = df.copy()
    num = out.select_dtypes(include=np.number).columns
    for c in num:
        q1, q3 = out[c].quantile(0.25), out[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        out[c] = out[c].clip(lower=low, upper=high)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera features simples (si existen las columnas).

    - BMI (IMC) si existen `weight` (kg) y `height` (cm).
    - `age_bin` con cortes etarios si existe `age`/`edad` (varios alias).

    Args:
        df: DataFrame limpio.

    Returns:
        DataFrame con nuevas columnas de features.
    """
    df = df.copy()
    low = {c.lower(): c for c in df.columns}

    # BMI = weight / (height/100)^2
    if "weight" in low and "height" in low:
        w, h = low["weight"], low["height"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["bmi"] = df[w] / ((df[h] / 100) ** 2)

    # age_bin si hay alguna variante de edad
    age_col = next((a for a in ["age", "edad", "Age", "Edad"] if a in df.columns), None)
    if age_col:
        bins = [-np.inf, 30, 45, 60, np.inf]
        labels = ["<=30", "31-45", "46-60", "60+"]
        df["age_bin"] = pd.cut(pd.to_numeric(df[age_col], errors="coerce"), bins=bins, labels=labels)

    return df


def prep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Nodo de preparación que devuelve (clean, features).

    Orquesta:
      - `basic_clean` → limpieza mínima y normalización.
      - `iqr_clip`   → clipping de outliers (IQR*3).
      - `engineer_features` → crea features simples.

    **Nota:** Este nodo no fija la columna objetivo (target); esa decisión se
    documenta en notebooks y/o en otros nodos/procesos de modelado.

    Args:
        df: DataFrame crudo.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (clean, features)
    """
    clean = basic_clean(df)
    clean = iqr_clip(clean, factor=3.0)
    feats = engineer_features(clean)
    return clean, feats
