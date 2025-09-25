import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _choose_target(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Devuelve el nombre de la columna target.
    Si preferred existe, la usa; si no, busca la primera columna binaria."""
    if preferred and preferred in df.columns:
        return preferred
    # buscar primera columna binaria (0/1 o 2 valores únicos)
    for col in df.columns:
        uniq = df[col].dropna().unique()
        if len(uniq) == 2:
            return col
    # si no hay binaria, usar la última columna como último recurso
    return df.columns[-1]

def _plot_target_bar(df: pd.DataFrame, target_col: str, title: str):
    counts = df[target_col].value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(target_col)
    ax.set_ylabel("frecuencia")
    fig.tight_layout()
    return fig

def _plot_hists(df: pd.DataFrame, title: str, max_cols: int = 9, bins: int = 20):
    num = df.select_dtypes(include=np.number)
    cols = num.columns[:max_cols]
    fig, axes = (plt.subplots(1,1, figsize=(6,4)) if len(cols)==0
                 else plt.subplots(int(np.ceil(len(cols)/3)), 3, figsize=(12, 3*int(np.ceil(len(cols)/3)))))
    if len(cols) == 0:
        ax = axes if isinstance(axes, plt.Axes) else axes[0]
        ax.text(0.5, 0.5, "No hay variables numéricas", ha="center", va="center"); ax.axis("off")
        return axes if isinstance(axes, plt.Figure) else plt.gcf()
    axes = np.atleast_1d(axes).ravel()
    for i, c in enumerate(cols):
        axes[i].hist(num[c].dropna(), bins=bins); axes[i].set_title(c)
    for j in range(i+1, len(axes)): axes[j].axis("off")
    fig = axes[0].figure
    fig.suptitle(title); fig.tight_layout()
    return fig

def report_for_dataset(df: pd.DataFrame, target_col: str, name: str):
    """Crea 2 figuras: barra del target + histogramas numéricos.
    Usa `target_col` si existe; si no, elige automáticamente una columna binaria."""
    target = _choose_target(df, preferred=target_col)
    fig_target = _plot_target_bar(df, target, f"{name} - dist. {target}")
    fig_hists = _plot_hists(df.drop(columns=[target], errors="ignore"), title=f"{name} - histogramas")
    return fig_target, fig_hists
