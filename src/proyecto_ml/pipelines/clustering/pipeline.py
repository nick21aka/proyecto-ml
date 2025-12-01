# src/proyecto_ml/pipelines/clustering/pipeline.py
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_clustering_models,
    create_cluster_plots,
    compute_elbow_curve,
    run_pca_analysis,
    run_tsne_embedding,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            # ---------------------------
            # 1) Modelos de clustering
            # ---------------------------
            node(
                func=run_clustering_models,
                inputs=["clustering_features", "params:clustering"],
                outputs=[
                    "clustering_results_csv",
                    "clustering_labels",
                    "clustering_silhouette",
                    "clustering_davies",
                    "clustering_calinski",
                ],
                name="run_clustering_models",
            ),

            # ---------------------------
            # 2) Gráficos de clusters (KMeans, DBSCAN, GMM, dendrograma)
            # ---------------------------
            node(
                func=create_cluster_plots,
                inputs=["clustering_features", "clustering_labels"],
                outputs=[
                    "clustering_kmeans_plot",
                    "clustering_dbscan_plot",
                    "clustering_gmm_plot",
                    "clustering_dendrogram_plot",  # dendrograma jerárquico
                ],
                name="create_cluster_plots",
            ),

            # ---------------------------
            # 3) Elbow Method (KMeans)
            # ---------------------------
            node(
                func=compute_elbow_curve,
                inputs="clustering_features",
                outputs=[
                    "clustering_elbow_data",
                    "clustering_elbow_plot",
                ],
                name="compute_elbow_curve",
            ),

            # ---------------------------
            # 4) PCA (varianza explicada, loadings, biplot)
            # ---------------------------
            node(
                func=run_pca_analysis,
                inputs="clustering_features",
                outputs=[
                    "pca_explained_variance",
                    "pca_loadings",
                    "pca_variance_plot",
                    "pca_biplot_plot",
                ],
                name="run_pca_analysis",
            ),

            # ---------------------------
            # 5) t-SNE 2D
            # ---------------------------
            node(
                func=run_tsne_embedding,
                inputs="clustering_features",
                outputs="tsne_2d_plot",
                name="run_tsne_embedding",
            ),
        ]
    )
