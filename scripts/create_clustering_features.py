import pandas as pd
from pathlib import Path

# =============================
# 1) Cargar el dataset base
# =============================
parquet_path = "data/03_primary/model_input_table.parquet"

print(f"Cargando datos desde: {parquet_path}")
df = pd.read_parquet(parquet_path)

# =============================
# 2) Seleccionar columnas numéricas
# =============================
numeric_cols = df.select_dtypes(include="number").columns
print(f"Columnas numéricas usadas como features ({len(numeric_cols)}):")
print(list(numeric_cols))

df_features = df[numeric_cols].copy()

# =============================
# 3) Asegurar carpeta y guardar parquet
# =============================
output_dir = Path("data/04_feature")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "clustering_features.parquet"
df_features.to_parquet(output_path, index=False)

print(f"\n✅ Archivo creado correctamente en: {output_path}")
