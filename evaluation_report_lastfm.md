# Spotify-Style Music Discovery Platform: Last.fm Offline Evaluation

This report evaluates the platform's recommendation and discovery system on processed Last.fm-style listening data using a candidate-aware benchmark.

## Dataset Summary

- Interaction count: 180,229
- User count: 973
- Track count: 123,788
- Evaluated user count: 925
- Minimum interaction threshold: 5
- Holdout strategy: most recent user-track interaction when timestamps exist; otherwise interaction strength fallback

## Dataset Coverage and Interaction Density

| Statistic | Value |
| --- | ---: |
| Interactions | 180,229 |
| Users | 973 |
| Tracks in interactions | 123,788 |
| Tracks in catalog | 123,788 |
| Avg interactions/user | 185.23 |
| Median interactions/user | 117.00 |
| Avg users/track | 1.46 |
| Median users/track | 1.00 |
| Tracks with >1 listener | 20.07% |
| Matrix density | 0.00149635 |
| Matrix sparsity | 0.99850365 |
| Evaluated users | 925 |
| Avg train interactions/evaluated user | 193.71 |

Collaborative overlap is **weak**: tracks average 1.46 listeners and 20.07% of tracks have more than one listener.

Matrix sparsity is **high** at 0.998504, so most possible user-track pairs are unobserved.

ALS **has limited latent-factor signal** under this split. Word2Vec-style embeddings **have useful sequence/co-occurrence signal** from the average user history length. Content methods **may remain competitive** when collaborative overlap or sequence structure is thin.

## Evaluation Setup

- Train/test split: deterministic user-level holdout.
- `holdout_count`: 1
- `K`: 10
- Candidate pool: top 1,000 catalog tracks by popularity, with each evaluated user's train and held-out tracks forced into that user's candidate list
- Metrics: Precision@10, Recall@10, and NDCG@10.
- Assumptions: processed Last.fm plays are treated as implicit user-track preference signals, and each held-out track is considered relevant for its user.

## Metrics Table

| Model | Precision@10 | Recall@10 | NDCG@10 | evaluated_users |
| --- | ---: | ---: | ---: | ---: |
| `content_only` | 0.023 | 0.114 | 0.099 | 925 |
| `hybrid` | 0.010 | 0.095 | 0.071 | 925 |
| `hybrid_plus_ALS` | 0.009 | 0.091 | 0.069 | 925 |
| `hybrid_plus_ALS_Word2Vec` | 0.009 | 0.085 | 0.065 | 925 |
| `ALS_only` | 0.000 | 0.003 | 0.003 | 925 |
| `collaborative_only` | 0.000 | 0.002 | 0.001 | 925 |
| `Word2Vec_similarity_only` | 0.000 | 0.003 | 0.001 | 925 |

## Interpretation

- ALS improved over the current collaborative baseline on NDCG@10.
- Word2Vec-style similarity did not improve over the content baseline on NDCG@10.
- `hybrid_plus_ALS` did not improve over the base hybrid model.
- `hybrid_plus_ALS_Word2Vec` did not improve over the base hybrid model.
- These results should be read alongside density and overlap statistics rather than treated as a final judgment of model quality.

Sparsity and overlap strongly affect collaborative methods. ALS needs enough repeated user-item structure to estimate stable latent factors, while Word2Vec-style embeddings need longer, consistently ordered listening sequences to learn reliable co-occurrence neighborhoods.

## Product Perspective

Offline metrics matter because they provide a repeatable way to compare rankers before user-facing experiments. Stronger architectures do not automatically improve metrics when their required signal is weak, sparse, or mismatched to the split strategy.

Production recommenders often combine multiple signals because each model family covers different failure modes: content similarity handles cold or sparse cases, collaborative models learn crowd behavior, embedding models capture context, and novelty/diversity controls keep recommendations from becoming repetitive.
