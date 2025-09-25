import logging, numpy as np, pandas as pd
logger = logging.getLogger(__name__)

def validate_not_empty(df: pd.DataFrame, name: str) -> pd.DataFrame:
    assert not df.empty, f"{name} está vacío"
    logger.info(f"{name} shape={df.shape}")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().replace({"": pd.NA})
    return df
