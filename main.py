from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from src.data_io import acquire_dataset, load_and_clean
from src.evaluation import first_hit_rank, ndcg_at_k, recall_at_k
from src.models import item_cf_scores, item_cosine_similarity, popularity_scores, stable_top_k
from src.pipeline import build_interaction_matrix, explicit_to_positive, temporal_leave_one_out

SEED = 42
TOP_K = 10


def main() -> None:
    started = time.perf_counter()
    np.random.seed(SEED)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    data_root, source = acquire_dataset(Path("data"))
    ratings, movies, raw_count = load_and_clean(data_root, source)
    positives = explicit_to_positive(ratings, threshold=4)
    train, test = temporal_leave_one_out(positives)
    user_ids = np.sort(test["user_id"].unique())
    item_ids = np.sort(np.union1d(movies["item_id"].unique(), positives["item_id"].unique()))
    matrix, user_to_index, _ = build_interaction_matrix(train, user_ids, item_ids)

    popularity = popularity_scores(matrix)
    similarity = item_cosine_similarity(matrix)
    title_by_id = movies.set_index("item_id")["title"].to_dict()
    test_target = test.set_index("user_id")["item_id"].astype(int).to_dict()
    train_counts = train.groupby("user_id").size().to_dict()
    train_item_counts = train.groupby("item_id").size().to_dict()
    recommendation_rows: list[dict] = []
    metric_rows: list[dict] = []

    for user_id in user_ids:
        user_index = user_to_index[int(user_id)]
        seen_indices = matrix.getrow(user_index).indices
        target = int(test_target[int(user_id)])
        method_scores = {
            "popularity": popularity,
            "item_cf": item_cf_scores(user_index, matrix, similarity),
        }
        for method, scores in method_scores.items():
            ranked = stable_top_k(scores, seen_indices, item_ids, TOP_K)
            recommended_ids = [item_id for item_id, _ in ranked]
            relevant = {target}
            metric_rows.append({
                "method": method,
                "user_id": int(user_id),
                "target_item_id": target,
                "recall_at_10": recall_at_k(recommended_ids, relevant, TOP_K),
                "ndcg_at_10": ndcg_at_k(recommended_ids, relevant, TOP_K),
                "hit_rank": first_hit_rank(recommended_ids, relevant, TOP_K),
                "train_positive_count": int(train_counts[int(user_id)]),
                "target_train_popularity": int(train_item_counts.get(target, 0)),
                "cold_target": int(target not in train_item_counts),
            })
            for rank, (item_id, score) in enumerate(ranked, start=1):
                recommendation_rows.append({
                    "method": method, "user_id": int(user_id), "rank": rank,
                    "item_id": item_id,
                    "title": title_by_id.get(item_id, f"item-{item_id}"),
                    "score": score, "is_test_item": int(item_id == target),
                })

    recommendations = pd.DataFrame(recommendation_rows)
    per_user = pd.DataFrame(metric_rows)
    summary = per_user.groupby("method", as_index=False).agg(
        users=("user_id", "size"),
        recall_at_10=("recall_at_10", "mean"),
        ndcg_at_10=("ndcg_at_10", "mean"),
    )
    failures = per_user[per_user["recall_at_10"] == 0].copy()
    failures["target_title"] = failures["target_item_id"].map(title_by_id)
    failures["failure_reason"] = np.select(
        [
            failures["cold_target"].eq(1),
            failures["train_positive_count"].le(3),
            failures["target_train_popularity"].le(2),
        ],
        ["物品冷启动", "用户历史过短", "目标物品极稀疏"],
        default="Top-10 未覆盖，需逐例分析",
    )
    recommendations.to_csv(output_dir / "user_recommendations.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output_dir / "failure_cases.csv", index=False, encoding="utf-8-sig")
    train.to_csv(output_dir / "train_positive.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / "test_positive.csv", index=False, encoding="utf-8-sig")

    sparsity = 1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    similarity_mib = (
        similarity.data.nbytes + similarity.indices.nbytes + similarity.indptr.nbytes
    ) / 1024**2
    print(f"python={platform.python_version()} numpy={np.__version__} pandas={pd.__version__}")
    print(f"scipy={scipy.__version__} scikit-learn={sklearn.__version__}")
    print(f"dataset_source={source} raw={raw_count} cleaned={len(ratings)} positives={len(positives)}")
    print(f"train={len(train)} test={len(test)} users={len(user_ids)} catalog={len(item_ids)}")
    print(f"matrix_shape={matrix.shape} nnz={matrix.nnz} sparsity={sparsity:.6f}")
    print(f"similarity_nnz={similarity.nnz} similarity_csr_mib={similarity_mib:.2f}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    print("outputs=5 CSV files in outputs/")


if __name__ == "__main__":
    main()
