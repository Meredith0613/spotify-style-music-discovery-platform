"""Word2Vec-inspired track embeddings from listening context.

The model uses deterministic co-occurrence, PPMI weighting, and SVD instead of
gensim. This keeps the project lightweight while still capturing the same broad
idea as Word2Vec: tracks that appear in similar listening contexts receive
nearby vectors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse.linalg import svds


@dataclass(slots=True)
class TrackEmbeddingArtifacts:
    """Store learned track embeddings aligned to track IDs."""

    track_ids: list[str]
    embedding_matrix: np.ndarray
    cooccurrence_matrix: csr_matrix


@dataclass(slots=True)
class TrackEmbeddingModel:
    """Train lightweight context embeddings from playlist or listening sequences."""

    embedding_dim: int = 16
    window_size: int = 2
    min_count: int = 1
    random_state: int = 42
    artifacts: TrackEmbeddingArtifacts | None = None
    _track_index_by_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def build_sequences(
        self,
        interactions: pd.DataFrame,
        *,
        user_column: str = "user_id",
        track_column: str = "track_id",
        timestamp_column: str | None = None,
        playlist_column: str | None = None,
        position_column: str | None = None,
    ) -> list[list[str]]:
        """Build ordered track sequences from playlist or user-history rows."""

        if interactions.empty or track_column not in interactions.columns:
            return []

        prepared = interactions.copy()
        prepared[track_column] = prepared[track_column].astype(str)
        prepared = prepared.loc[prepared[track_column].str.strip() != ""].copy()
        if prepared.empty:
            return []

        group_column = playlist_column if playlist_column and playlist_column in prepared.columns else user_column
        if group_column not in prepared.columns:
            return [prepared[track_column].astype(str).tolist()]

        sort_columns = self._resolve_sort_columns(
            prepared,
            timestamp_column=timestamp_column,
            position_column=position_column,
        )
        sequences: list[list[str]] = []
        for _, group_frame in prepared.groupby(group_column, sort=True):
            ordered_frame = group_frame.sort_values(sort_columns, kind="stable") if sort_columns else group_frame
            sequence = ordered_frame[track_column].astype(str).tolist()
            if sequence:
                sequences.append(sequence)
        return sequences

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_column: str = "user_id",
        track_column: str = "track_id",
        timestamp_column: str | None = None,
        playlist_column: str | None = None,
        position_column: str | None = None,
    ) -> "TrackEmbeddingModel":
        """Fit embeddings from an interaction table."""

        sequences = self.build_sequences(
            interactions,
            user_column=user_column,
            track_column=track_column,
            timestamp_column=timestamp_column,
            playlist_column=playlist_column,
            position_column=position_column,
        )
        return self.fit_sequences(sequences)

    def fit_sequences(self, sequences: list[list[str]]) -> "TrackEmbeddingModel":
        """Fit embeddings from pre-built track sequences."""

        vocabulary = self._build_vocabulary(sequences)
        if not vocabulary:
            self.artifacts = TrackEmbeddingArtifacts(
                track_ids=[],
                embedding_matrix=np.empty((0, 0)),
                cooccurrence_matrix=csr_matrix((0, 0)),
            )
            self._track_index_by_id = {}
            return self

        cooccurrence = self._build_cooccurrence_matrix(sequences, vocabulary)
        embedding_matrix = self._factorize_ppmi(cooccurrence)
        self.artifacts = TrackEmbeddingArtifacts(
            track_ids=vocabulary,
            embedding_matrix=embedding_matrix,
            cooccurrence_matrix=cooccurrence,
        )
        self._track_index_by_id = {track_id: index for index, track_id in enumerate(vocabulary)}
        return self

    def get_track_vector(self, track_id: str) -> np.ndarray | None:
        """Return a learned track vector or None when the track is unseen."""

        if self.artifacts is None:
            return None
        track_index = self._track_index_by_id.get(str(track_id))
        if track_index is None:
            return None
        return self.artifacts.embedding_matrix[track_index]

    def similar_tracks(self, track_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the most contextually similar tracks to a known track."""

        if k <= 0 or self.artifacts is None:
            return []
        query_vector = self.get_track_vector(track_id)
        if query_vector is None:
            return []

        scores: list[tuple[str, float]] = []
        for candidate_id, candidate_vector in zip(self.artifacts.track_ids, self.artifacts.embedding_matrix):
            if candidate_id == str(track_id):
                continue
            scores.append(
                (
                    candidate_id,
                    self._combined_context_similarity(str(track_id), candidate_id, query_vector, candidate_vector),
                )
            )
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:k]

    def score_candidate_similarity(
        self,
        seed_track_ids: list[str],
        candidate_track_ids: list[str],
    ) -> dict[str, float]:
        """Score candidates by cosine similarity to the average seed context."""

        seed_vectors = [
            vector
            for seed_track_id in seed_track_ids
            if (vector := self.get_track_vector(seed_track_id)) is not None
        ]
        if not seed_vectors:
            return {str(track_id): 0.0 for track_id in candidate_track_ids}

        seed_profile = np.mean(np.vstack(seed_vectors), axis=0)
        normalized_candidate_ids = [str(track_id) for track_id in candidate_track_ids]
        candidate_indices = [self._track_index_by_id.get(track_id) for track_id in normalized_candidate_ids]
        scores = {track_id: 0.0 for track_id in normalized_candidate_ids}
        if self.artifacts is None:
            return scores

        known_positions = [
            position
            for position, track_index in enumerate(candidate_indices)
            if track_index is not None
        ]
        if not known_positions:
            return scores

        known_indices = [candidate_indices[position] for position in known_positions]
        known_track_ids = [normalized_candidate_ids[position] for position in known_positions]
        candidate_vectors = self.artifacts.embedding_matrix[known_indices]
        embedding_scores = self._cosine_similarity_many(seed_profile, candidate_vectors)
        direct_context_scores = self._average_direct_context_scores(seed_track_ids, known_indices)
        final_scores = (0.65 * direct_context_scores) + (0.35 * embedding_scores)
        scores.update(
            {
                track_id: float(score)
                for track_id, score in zip(known_track_ids, final_scores)
            }
        )
        return scores

    def _resolve_sort_columns(
        self,
        interactions: pd.DataFrame,
        *,
        timestamp_column: str | None,
        position_column: str | None,
    ) -> list[str]:
        """Resolve stable sequence-ordering columns from available metadata."""

        sort_columns: list[str] = []
        if position_column and position_column in interactions.columns:
            sort_columns.append(position_column)
        if timestamp_column and timestamp_column in interactions.columns:
            sort_columns.append(timestamp_column)
        for fallback_column in ("timestamp", "played_at", "first_timestamp", "last_timestamp"):
            if not sort_columns and fallback_column in interactions.columns:
                sort_columns.append(fallback_column)
        return sort_columns

    def _build_vocabulary(self, sequences: list[list[str]]) -> list[str]:
        """Return deterministic track vocabulary after min-count filtering."""

        counts = Counter(track_id for sequence in sequences for track_id in sequence)
        return sorted(
            track_id
            for track_id, count in counts.items()
            if count >= max(int(self.min_count), 1)
        )

    def _build_cooccurrence_matrix(self, sequences: list[list[str]], vocabulary: list[str]) -> csr_matrix:
        """Build a weighted symmetric co-occurrence matrix."""

        index_by_track_id = {track_id: index for index, track_id in enumerate(vocabulary)}
        pair_counts: dict[tuple[int, int], float] = {}
        for sequence in sequences:
            filtered_sequence = [track_id for track_id in sequence if track_id in index_by_track_id]
            for left_position, left_track_id in enumerate(filtered_sequence):
                left_index = index_by_track_id[left_track_id]
                start = max(0, left_position - self.window_size)
                stop = min(len(filtered_sequence), left_position + self.window_size + 1)
                for right_position in range(start, stop):
                    if right_position == left_position:
                        continue
                    distance = abs(right_position - left_position)
                    right_index = index_by_track_id[filtered_sequence[right_position]]
                    pair = (left_index, right_index)
                    pair_counts[pair] = pair_counts.get(pair, 0.0) + (1.0 / max(distance, 1))
        if not pair_counts:
            return csr_matrix((len(vocabulary), len(vocabulary)), dtype=float)

        rows = [pair[0] for pair in pair_counts]
        columns = [pair[1] for pair in pair_counts]
        values = list(pair_counts.values())
        return coo_matrix(
            (values, (rows, columns)),
            shape=(len(vocabulary), len(vocabulary)),
            dtype=float,
        ).tocsr()

    def _factorize_ppmi(self, cooccurrence: csr_matrix) -> np.ndarray:
        """Convert co-occurrence counts to PPMI and factorize with SVD."""

        if cooccurrence.size == 0:
            return np.empty((0, 0))
        if cooccurrence.nnz == 0:
            return self._fallback_embeddings(cooccurrence.shape[0])

        total_count = float(cooccurrence.sum())
        row_totals = np.asarray(cooccurrence.sum(axis=1)).ravel()
        column_totals = np.asarray(cooccurrence.sum(axis=0)).ravel()
        coo = cooccurrence.tocoo()
        expected_values = (row_totals[coo.row] * column_totals[coo.col]) / max(total_count, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            ppmi_values = np.maximum(np.log(coo.data / np.maximum(expected_values, 1e-12)), 0.0)
        positive_mask = ppmi_values > 0.0
        if not np.any(positive_mask):
            return self._fallback_embeddings(cooccurrence.shape[0])

        ppmi = coo_matrix(
            (ppmi_values[positive_mask], (coo.row[positive_mask], coo.col[positive_mask])),
            shape=cooccurrence.shape,
            dtype=float,
        ).tocsr()
        dimension = max(1, min(int(self.embedding_dim), min(ppmi.shape) - 1))
        if ppmi.shape[0] <= 200:
            dense_ppmi = ppmi.toarray()
            left_vectors, singular_values, _ = np.linalg.svd(dense_ppmi, full_matrices=False)
            dimension = max(1, min(int(self.embedding_dim), left_vectors.shape[1]))
            return left_vectors[:, :dimension] * np.sqrt(singular_values[:dimension])

        left_vectors, singular_values, _ = svds(ppmi, k=dimension)
        order = np.argsort(singular_values)[::-1]
        return left_vectors[:, order] * np.sqrt(singular_values[order])

    def _fallback_embeddings(self, track_count: int) -> np.ndarray:
        """Create deterministic fallback vectors when co-occurrence has no signal."""

        dimension = max(1, int(self.embedding_dim))
        rng = np.random.default_rng(self.random_state)
        return rng.normal(scale=0.01, size=(track_count, dimension))

    def _combined_context_similarity(
        self,
        left_track_id: str,
        right_track_id: str,
        left_vector: np.ndarray,
        right_vector: np.ndarray,
    ) -> float:
        """Blend direct co-occurrence and embedding cosine similarity."""

        direct_context_score = self._direct_context_score(left_track_id, right_track_id)
        embedding_score = self._cosine_similarity(left_vector, right_vector)
        return (0.65 * direct_context_score) + (0.35 * embedding_score)

    def _average_direct_context_score(self, seed_track_ids: list[str], candidate_track_id: str) -> float:
        """Average direct co-occurrence strength from seeds to one candidate."""

        direct_scores = [
            self._direct_context_score(seed_track_id, candidate_track_id)
            for seed_track_id in seed_track_ids
        ]
        if not direct_scores:
            return 0.0
        return float(sum(direct_scores) / len(direct_scores))

    def _average_direct_context_scores(self, seed_track_ids: list[str], candidate_indices: list[int]) -> np.ndarray:
        """Vectorized direct co-occurrence scores from seeds to candidates."""

        if self.artifacts is None or not candidate_indices:
            return np.zeros((len(candidate_indices),), dtype=float)
        seed_indices = [
            self._track_index_by_id[seed_track_id]
            for seed_track_id in seed_track_ids
            if seed_track_id in self._track_index_by_id
        ]
        if not seed_indices:
            return np.zeros((len(candidate_indices),), dtype=float)

        direct_scores = np.zeros((len(candidate_indices),), dtype=float)
        contributing_seed_count = 0
        for seed_index in seed_indices:
            row = self.artifacts.cooccurrence_matrix.getrow(seed_index)
            if row.nnz == 0:
                continue
            row_max = float(row.data.max())
            if row_max == 0.0:
                continue
            direct_scores += np.asarray(row[:, candidate_indices].toarray()).ravel() / row_max
            contributing_seed_count += 1
        if contributing_seed_count == 0:
            return direct_scores
        return direct_scores / contributing_seed_count

    def _direct_context_score(self, left_track_id: str, right_track_id: str) -> float:
        """Return normalized direct co-occurrence between two known tracks."""

        if self.artifacts is None:
            return 0.0
        left_index = self._track_index_by_id.get(str(left_track_id))
        right_index = self._track_index_by_id.get(str(right_track_id))
        if left_index is None or right_index is None:
            return 0.0
        row = self.artifacts.cooccurrence_matrix.getrow(left_index)
        row_max = float(row.data.max()) if row.nnz else 0.0
        if row_max == 0.0:
            return 0.0
        return float(row[0, right_index] / row_max)

    def _cosine_similarity(self, left_vector: np.ndarray, right_vector: np.ndarray) -> float:
        """Return cosine similarity with safe zero-vector handling."""

        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(left_vector, right_vector) / denominator)

    def _cosine_similarity_many(self, query_vector: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity between one query and many candidates."""

        query_norm = float(np.linalg.norm(query_vector))
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        denominator = candidate_norms * query_norm
        raw_scores = candidate_vectors @ query_vector
        return np.divide(
            raw_scores,
            denominator,
            out=np.zeros_like(raw_scores, dtype=float),
            where=denominator != 0.0,
        )
