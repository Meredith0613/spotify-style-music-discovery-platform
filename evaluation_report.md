# Spotify-Style Music Discovery Platform: Offline Evaluation Report

## 1. Setup

The Spotify-Style Music Discovery Platform was evaluated on the synthetic demo dataset that powers the Streamlit discovery experience. The dataset contains a small demo catalog and a compact set of user interaction logs designed to exercise the hybrid ranking stack in a controlled way.

The offline evaluation uses a deterministic user-level holdout split:

- Each evaluated user contributes exactly 1 held-out interaction to the test set.
- The remaining interactions for that user stay in the training set.
- Because the synthetic interaction logs do not include reliable event timestamps, `interaction_strength` is used to order interactions for holdout selection instead of recency.
- Metrics are computed at `K=3`.

Evaluated models:

- `content_only`
- `collaborative_only`
- `hybrid`
- `ALS_only`
- `Word2Vec_similarity_only`
- `hybrid_plus_ALS`
- `hybrid_plus_ALS_Word2Vec`
- `hybrid_spotify_reranked`

Key assumptions:

- The synthetic demo interaction logs are sufficient for a small offline sanity check.
- A single held-out interaction per user is enough to compare ranking behavior, but not enough to estimate production performance.
- `interaction_strength` is treated as the best available proxy for preference salience in the absence of timestamps.
- Spotify-aware reranking is evaluated as a post-ranking adjustment on top of the hybrid recommender, not as a separate recommender.

## 2. Metrics Table

| Model | Precision@3 | Recall@3 | NDCG@3 |
| --- | ---: | ---: | ---: |
| `content_only` | 0.361 | 1.000 | 0.794 |
| `hybrid` | 0.333 | 1.000 | 0.671 |
| `hybrid_plus_ALS` | 0.333 | 1.000 | 0.671 |
| `hybrid_plus_ALS_Word2Vec` | 0.333 | 1.000 | 0.671 |
| `hybrid_spotify_reranked` | 0.333 | 1.000 | 0.671 |
| `collaborative_only` | 0.361 | 0.833 | 0.565 |
| `Word2Vec_similarity_only` | 0.278 | 0.833 | 0.565 |
| `ALS_only` | 0.111 | 0.333 | 0.250 |

## 3. Key Findings

`content_only` performs best on this split because the synthetic users are intentionally built around strong clustered preferences. When the held-out item belongs to the same tight taste cluster as the remaining user history, content similarity is a very strong retrieval signal.

`collaborative_only` is weaker because the interaction matrix is tiny and sparse. With only a few users and limited overlap between their interactions, item-item collaborative signals are less stable and less informative than content similarity.

The `hybrid` model recovers all held-out items within the top 3, which is why Recall@3 remains perfect, but its ranking order is slightly worse than `content_only`. That is consistent with a hybrid system that balances multiple objectives rather than optimizing only for nearest-neighbor retrieval.

`hybrid_spotify_reranked` is neutral on this evaluation. The reranking layer is intentionally lightweight and explainable, and on a small synthetic dataset it does not move enough items to change the aggregate top-3 metrics.

## ALS and Word2Vec Findings

ALS and Word2Vec-style embeddings did not outperform the content-based baseline on the synthetic evaluation dataset.

This outcome is expected because the synthetic dataset contains a very small number of users and limited interaction overlap, reducing the ability of latent-factor and embedding-based methods to learn meaningful collaborative structure.

The content-based model performed best because synthetic user profiles were intentionally constructed around strong taste clusters, making content similarity highly predictive of held-out items.

The ALS and embedding pipelines were retained because they are expected to provide greater benefit on larger real-world listening datasets with richer user-item interaction patterns and track co-occurrence structure.

### Key Takeaways

- Strong content signals can outperform collaborative methods on tiny datasets.
- ALS benefits from larger user-item matrices and richer interaction overlap.
- Word2Vec-style embeddings benefit from longer listening sequences and stronger co-occurrence structure.
- Real-world evaluation is required before judging collaborative methods.

## 4. Product vs Metric Discussion

Offline ranking metrics measure ranking quality under a controlled train/test setup. In particular, NDCG@K captures not just whether the relevant item appears in the top-K list, but how high it appears in that ranked list.

Precision@K and Recall@K measure how many relevant items are surfaced in the top-K recommendations. Precision@K emphasizes recommendation efficiency, while Recall@K emphasizes coverage of relevant items.

Why the hybrid ranking stack is not best offline:

- The hybrid ranker is designed to balance collaborative, content, novelty, popularity, and discovery signals.
- That means it is not optimized solely for maximizing held-out ranking metrics on a tiny synthetic split.
- In product terms, a hybrid system may produce a healthier mix of familiarity and exploration even when a pure content model scores slightly better offline.

Why the reranking effect is not visible here:

- The evaluation dataset is very small.
- The reranking layer is intentionally conservative so that it nudges rankings instead of overpowering the base recommender.
- On this split, those adjustments are too small to change the aggregate top-K metrics.

Why content dominates:

- The synthetic users have strong clustered preferences.
- The demo catalog is small and semantically tight.
- Under those conditions, content similarity is often enough to recover the held-out item directly.

Offline metrics are necessary but not sufficient; real systems require online evaluation, such as A/B testing, to measure user impact, satisfaction, and behavioral lift in production.

## 5. Limitations

- The evaluation uses a synthetic dataset rather than real-world interaction logs.
- Only 6 users are evaluated.
- Each user contributes only 1 held-out interaction.
- The item catalog is small.
- The setup is useful for comparative sanity checks, but it is not representative of real-world recommendation scale or user behavior complexity.

## 6. Next Evaluation Improvements

- Evaluate on a larger dataset with more users and denser interaction histories.
- Report metrics at multiple K values, such as `K=5` and `K=10`, in addition to `K=3`.
- Add diversity and novelty metrics to better reflect product goals beyond relevance alone.
- Run an ablation study to isolate the impact of collaborative, content, novelty, popularity, discovery, and reranking components.
- Add visual summaries, including the NDCG bar chart generated by `evaluation_plot.py`, to make model comparisons easier to communicate.
