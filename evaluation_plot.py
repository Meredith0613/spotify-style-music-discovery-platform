"""Simple offline evaluation visualization for recruiter-facing reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    model_names = [
        "content_only",
        "hybrid",
        "hybrid_spotify_reranked",
        "collaborative_only",
    ]
    ndcg_scores = [0.794, 0.671, 0.671, 0.565]

    output_path = Path("artifacts") / "evaluation_ndcg_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.bar(model_names, ndcg_scores, color=["#1DB954", "#4C78A8", "#72B7B2", "#F58518"])
    plt.title("Offline Evaluation: NDCG@3 by Model")
    plt.ylabel("NDCG@3")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
