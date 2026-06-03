"""Optional projection and clustering helpers for feature exploration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ProjectionBuilder:
    """Provide optional dimensionality reduction and clustering helpers.

    This helper is intentionally separate from the main feature builder so
    exploratory visualization logic does not leak into production feature code.
    """

    def project_tsne(
        self,
        feature_matrix: np.ndarray,
        n_components: int = 2,
        random_state: int = 42,
    ) -> np.ndarray:
        """Project a feature matrix into a lower-dimensional t-SNE space.

        Args:
            feature_matrix: Standardized feature matrix for tracks.
            n_components: Number of output dimensions to produce.
            random_state: Seed used for reproducible projections.

        Returns:
            Low-dimensional coordinates for each track.
        """

        if feature_matrix.size == 0:
            return np.empty((0, n_components))

        try:
            from sklearn.manifold import TSNE
        except ImportError as error:  # pragma: no cover - optional dependency path
            raise ImportError(
                "t-SNE projection requires scikit-learn. Install it to use ProjectionBuilder.project_tsne."
            ) from error

        # t-SNE is intended for EDA and visualization rather than the main
        # recommendation pipeline, so it remains an optional helper.
        model = TSNE(n_components=n_components, random_state=random_state, init="pca")
        return model.fit_transform(feature_matrix)

    def project_umap(
        self,
        feature_matrix: np.ndarray,
        n_components: int = 2,
        random_state: int = 42,
    ) -> np.ndarray:
        """Project a feature matrix into a lower-dimensional UMAP space.

        Args:
            feature_matrix: Standardized feature matrix for tracks.
            n_components: Number of output dimensions to produce.
            random_state: Seed used for reproducible projections.

        Returns:
            Low-dimensional coordinates for each track.
        """

        if feature_matrix.size == 0:
            return np.empty((0, n_components))

        try:
            import umap
        except ImportError as error:  # pragma: no cover - optional dependency path
            raise ImportError(
                "UMAP projection requires umap-learn. Install it to use ProjectionBuilder.project_umap."
            ) from error

        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(feature_matrix)

    def cluster_kmeans(
        self,
        feature_matrix: np.ndarray,
        n_clusters: int = 8,
        random_state: int = 42,
    ) -> np.ndarray:
        """Cluster tracks in feature space using k-means.

        Args:
            feature_matrix: Standardized feature matrix for tracks.
            n_clusters: Number of clusters to assign.
            random_state: Seed used for reproducible clustering.

        Returns:
            Integer cluster labels for each track.
        """

        if feature_matrix.size == 0:
            return np.empty((0,), dtype=int)

        try:
            from sklearn.cluster import KMeans
        except ImportError as error:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Clustering requires scikit-learn. Install it to use ProjectionBuilder.cluster_kmeans."
            ) from error

        # Clustering is most useful for discovery analysis, playlist shaping,
        # and visualization rather than the first-pass recommender itself.
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        return model.fit_predict(feature_matrix)
