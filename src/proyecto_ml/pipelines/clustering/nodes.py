from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ======================================================
# 1. ENTRENAR MODELOS DE CLUSTERING + MÉTRICAS
# ======================================================
def _compute_metrics(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Calcula Silhouette, Davies-Bouldin y Calinski-Harabasz.
    Si no hay al menos 2 clusters válidos, devuelve NaN.
    """
    # clusters válidos (ignorando ruido -1, útil para DBSCAN)
    mask_valid = labels != -1
    unique_labels = np.unique(labels[mask_valid])

    if unique_labels.size < 2:
        return np.nan, np.nan, np.nan

    X_valid = X[mask_valid]
    labels_valid = labels[mask_valid]

    sil = float(silhouette_score(X_valid, labels_valid))
    dav = float(davies_bouldin_score(X_valid, labels_valid))
    cal = float(calinski_harabasz_score(X_valid, labels_valid))
    return sil, dav, cal


def run_clustering_models(
    data: pd.DataFrame,
    params: Dict[str, Any],
):
    """Ejecuta KMeans, DBSCAN, GMM y Jerárquico.
    Devuelve:
      - DataFrame con métricas por algoritmo
      - DataFrame con etiquetas de cluster
      - dicts de métricas (para JSON)
    """

    # 1) Tomamos solo columnas numéricas
    X = data.select_dtypes(include=[np.number]).to_numpy()

    # 2) Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    labels_dict: Dict[str, np.ndarray] = {}

    # Parámetros generales
    random_state = params.get("random_state", 42)

    # ---------------- KMEANS ----------------
    km_cfg = params.get("kmeans", {})
    k_kmeans = km_cfg.get("n_clusters", 2)

    kmeans = KMeans(n_clusters=k_kmeans, random_state=random_state, n_init="auto")
    labels_km = kmeans.fit_predict(X_scaled)
    sil, dav, cal = _compute_metrics(X_scaled, labels_km)

    results.append(
        {
            "model": "kmeans",
            "n_clusters": int(len(np.unique(labels_km[labels_km != -1]))),
            "silhouette": sil,
            "davies_bouldin": dav,
            "calinski_harabasz": cal,
        }
    )
    labels_dict["kmeans"] = labels_km

    # ---------------- DBSCAN ----------------
    db_cfg = params.get("dbscan", {})
    eps = db_cfg.get("eps", 0.5)
    min_samples = db_cfg.get("min_samples", 5)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = dbscan.fit_predict(X_scaled)
    sil, dav, cal = _compute_metrics(X_scaled, labels_db)

    results.append(
        {
            "model": "dbscan",
            "n_clusters": int(len(np.unique(labels_db[labels_db != -1]))),
            "silhouette": sil,
            "davies_bouldin": dav,
            "calinski_harabasz": cal,
        }
    )
    labels_dict["dbscan"] = labels_db

    # ---------------- GMM ----------------
    gmm_cfg = params.get("gmm", {})
    k_gmm = gmm_cfg.get("n_components", 2)

    gmm = GaussianMixture(n_components=k_gmm, random_state=random_state)
    labels_gmm = gmm.fit_predict(X_scaled)
    sil, dav, cal = _compute_metrics(X_scaled, labels_gmm)

    results.append(
        {
            "model": "gmm",
            "n_clusters": int(len(np.unique(labels_gmm[labels_gmm != -1]))),
            "silhouette": sil,
            "davies_bouldin": dav,
            "calinski_harabasz": cal,
        }
    )
    labels_dict["gmm"] = labels_gmm

    # ---------------- JERÁRQUICO ----------------
    h_cfg = params.get("hierarchical", {})
    k_hier = h_cfg.get("n_clusters", 2)
    linkage = h_cfg.get("linkage", "ward")

    hier = AgglomerativeClustering(n_clusters=k_hier, linkage=linkage)
    labels_hier = hier.fit_predict(X_scaled)
    sil, dav, cal = _compute_metrics(X_scaled, labels_hier)

    results.append(
        {
            "model": "hierarchical",
            "n_clusters": int(len(np.unique(labels_hier[labels_hier != -1]))),
            "silhouette": sil,
            "davies_bouldin": dav,
            "calinski_harabasz": cal,
        }
    )
    labels_dict["hierarchical"] = labels_hier

    # ---------------- ARMAR SALIDAS ----------------
    df_results = pd.DataFrame(results)

    # DataFrame de etiquetas para usar en los plots
    df_labels = pd.DataFrame(labels_dict)

    # Diccionarios para JSON (puedes usarlos en el informe)
    sil_dict = {row["model"]: row["silhouette"] for _, row in df_results.iterrows()}
    dav_dict = {row["model"]: row["davies_bouldin"] for _, row in df_results.iterrows()}
    cal_dict = {
        row["model"]: row["calinski_harabasz"] for _, row in df_results.iterrows()
    }

    return df_results, df_labels, sil_dict, dav_dict, cal_dict


# ======================================================
# 2. PLOTS (KMEANS, DBSCAN, GMM, DENDROGRAMA)
# ======================================================
from scipy.cluster.hierarchy import linkage, dendrogram


def create_cluster_plots(
    data: pd.DataFrame,
    labels_df: pd.DataFrame,
):
    """Genera:
      - Scatter PCA 2D para KMeans
      - Scatter PCA 2D para DBSCAN (o mensaje si no hay clusters válidos)
      - Scatter PCA 2D para GMM
      - Dendrograma (jerárquico)
    """

    X = data.select_dtypes(include=[np.number]).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ---------- KMEANS ----------
    fig_km, ax_km = plt.subplots(figsize=(6, 5))
    ax_km.set_title("KMEANS - PCA 2D")
    ax_km.set_xlabel("PC1")
    ax_km.set_ylabel("PC2")

    labels_km = labels_df["kmeans"].to_numpy()
    for lab in np.unique(labels_km):
        mask = labels_km == lab
        ax_km.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, label=f"Cluster {lab}")
    ax_km.legend(title="Clusters")
    fig_km.tight_layout()

    # ---------- DBSCAN ----------
    fig_db, ax_db = plt.subplots(figsize=(6, 5))
    labels_db = labels_df["dbscan"].to_numpy()
    unique_db = np.unique(labels_db[labels_db != -1])

    if unique_db.size < 2:
        ax_db.set_title("DBSCAN - sin clusters válidos")
        ax_db.set_xlabel("PC1")
        ax_db.set_ylabel("PC2")
    else:
        ax_db.set_title("DBSCAN - PCA 2D")
        ax_db.set_xlabel("PC1")
        ax_db.set_ylabel("PC2")
        for lab in np.unique(labels_db):
            mask = labels_db == lab
            name = "Ruido" if lab == -1 else f"Cluster {lab}"
            ax_db.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, label=name)
        ax_db.legend(title="Clusters")

    fig_db.tight_layout()

    # ---------- GMM ----------
    fig_gmm, ax_gmm = plt.subplots(figsize=(6, 5))
    ax_gmm.set_title("GMM - PCA 2D")
    ax_gmm.set_xlabel("PC1")
    ax_gmm.set_ylabel("PC2")

    labels_gmm = labels_df["gmm"].to_numpy()
    for lab in np.unique(labels_gmm):
        mask = labels_gmm == lab
        ax_gmm.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, label=f"Cluster {lab}")
    ax_gmm.legend(title="Clusters")
    fig_gmm.tight_layout()

    # ---------- DENDROGRAMA (JERÁRQUICO) ----------
    # Para que no sea gigante, muestreamos hasta 500 puntos
    max_samples = 500
    if X_scaled.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(
            X_scaled.shape[0], size=max_samples, replace=False
        )
        X_for_dendro = X_scaled[idx]
    else:
        X_for_dendro = X_scaled

    Z = linkage(X_for_dendro, method="ward")

    fig_dendro, ax_d = plt.subplots(figsize=(8, 5))
    ax_d.set_title("Dendrograma - Clustering jerárquico (ward)")
    dendrogram(Z, ax=ax_d, truncate_mode="level", p=5, no_labels=True)
    ax_d.set_xlabel("Observaciones")
    ax_d.set_ylabel("Distancia")
    fig_dendro.tight_layout()

    return fig_km, fig_db, fig_gmm, fig_dendro
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def compute_elbow_curve(features: pd.DataFrame):
    """
    Calcula la curva del codo (Elbow Method) usando K-Means.
    Devuelve:
      - elbow_data: DataFrame con columnas ['k', 'inertia']
      - fig: Figura de Matplotlib con el gráfico del codo
    """

    # 1) Nos quedamos solo con columnas numéricas
    X = features.select_dtypes(include=[np.number]).dropna()

    # 2) Definimos el rango de k
    ks = list(range(2, 11))
    inertias = []

    for k in ks:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto",
        )
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # 3) DataFrame con los resultados
    elbow_data = pd.DataFrame(
        {
            "k": ks,
            "inertia": inertias,
        }
    )

    # 4) Creamos la figura del gráfico
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia (Suma de distancias cuadradas)")
    ax.set_title("Elbow Method - KMeans")
    ax.grid(True)

    # 5) IMPORTANTE: devolvemos DOS cosas (data + figura)
    return elbow_data, fig


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_pca_analysis(features: pd.DataFrame, n_components: int = 2):
    """
    Ejecuta PCA sobre las features numéricas y genera:
    - Varianza explicada (JSON-serializable)
    - Loadings (DataFrame)
    - Scree plot (fig_var)
    - Biplot PC1 vs PC2 (fig_biplot)
    """

    # 🔹 1) Quedarse SOLO con columnas numéricas y sin NaN
    numeric_features = features.select_dtypes(include=[np.number]).dropna()

    if numeric_features.empty:
        raise ValueError("No hay columnas numéricas válidas para aplicar PCA.")

    # Ajustar n_components si es mayor al número de columnas
    max_components = min(n_components, numeric_features.shape[1])
    if max_components < n_components:
        n_components = max_components

    # 🔹 2) Ajustar PCA sobre las columnas numéricas
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(numeric_features.values)

    # 🔹 Varianza explicada
    explained_variance = pca.explained_variance_ratio_
    pca_explained_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(explained_variance))],
        "explained_variance_ratio": explained_variance
    })

    # 🔹 Loadings: variables originales vs componentes
    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric_features.columns,  # columnas numéricas
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings.reset_index(inplace=True)
    loadings.rename(columns={"index": "feature"}, inplace=True)

    # ----- Gráfico 1: Scree plot (varianza explicada) -----
    fig_var, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(
        x=pca_explained_df["PC"],
        height=pca_explained_df["explained_variance_ratio"],
    )
    ax1.set_xlabel("Componente Principal")
    ax1.set_ylabel("Proporción de varianza explicada")
    ax1.set_title("PCA - Varianza explicada por componente")
    ax1.grid(True, axis="y")

    # ----- Gráfico 2: Biplot (PC1 vs PC2 + vectores de loadings) -----
    if n_components >= 2:
        pc1 = pcs[:, 0]
        pc2 = pcs[:, 1]

        fig_biplot, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(pc1, pc2, alpha=0.4, s=10)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("PCA - Biplot (PC1 vs PC2)")
        ax2.grid(True)

        max_features_to_plot = min(10, loadings.shape[0])
        pc1_max = max(abs(pc1))
        pc2_max = max(abs(pc2))

        for i in range(max_features_to_plot):
            feature = loadings.loc[i, "feature"]
            loading_pc1 = loadings.loc[i, "PC1"]
            loading_pc2 = loadings.loc[i, "PC2"]
            ax2.arrow(
                0,
                0,
                loading_pc1 * pc1_max,
                loading_pc2 * pc2_max,
                color="red",
                alpha=0.7,
                head_width=0.5,
                length_includes_head=True,
            )
            ax2.text(
                loading_pc1 * pc1_max * 1.05,
                loading_pc2 * pc2_max * 1.05,
                feature,
                color="red",
                fontsize=8,
            )
    else:
        fig_biplot, ax2 = plt.subplots(figsize=(6, 4))
        ax2.set_title("PCA - Biplot no disponible (menos de 2 componentes)")
        ax2.axis("off")

    # 👇 AQUÍ está el cambio importante:
    #  - Primer output → JSON-serializable (lista de diccionarios)
    #  - Segundo output → DataFrame (para CSVDataset)
    pca_explained_for_json = pca_explained_df.to_dict(orient="records")

    return pca_explained_for_json, loadings, fig_var, fig_biplot


def run_tsne_embedding(
    features: pd.DataFrame,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000
) -> plt.Figure:
    """
    Calcula un embedding 2D usando t-SNE y genera un scatter plot.
    Por rendimiento, si hay muchas filas, se toma una muestra.
    Solo se usan columnas numéricas.
    """
    # 🔹 1) Quedarse SOLO con variables numéricas y sin NaN
    numeric_features = features.select_dtypes(include=[np.number]).dropna()

    # 🔹 2) Submuestreo para que t-SNE no reviente el tiempo
    max_samples = 2000
    if len(numeric_features) > max_samples:
        sample = numeric_features.sample(n=max_samples, random_state=42)
    else:
        sample = numeric_features

    # (Opcional: ajustar perplexity si hay pocas filas)
    n_samples = len(sample)
    if n_samples <= 3 * perplexity:
        # t-SNE exige n_samples > 3 * perplexity
        perplexity = max(5.0, (n_samples - 1) / 3)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
        init="pca"
    )
    tsne_coords = tsne.fit_transform(sample.values)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], s=10, alpha=0.5)
    ax.set_title("t-SNE - Embedding 2D")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True)

    return fig
