from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def popularity_scores(train_matrix: csr_matrix) -> np.ndarray:
    """仅由训练矩阵统计每个物品的正反馈次数。"""
    return np.asarray(
        train_matrix.sum(axis=0)
    ).ravel().astype(np.float32)


def item_cosine_similarity(train_matrix: csr_matrix) -> csr_matrix:
    """构造稀疏物品余弦相似度矩阵，并清除对角线。"""
    item_user = train_matrix.T.tocsr()
    item_user = normalize(item_user, norm="l2", axis=1, copy=True)
    similarity = (item_user @ item_user.T).tocsr()
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    return similarity


def item_cf_scores(
    user_index: int, train_matrix: csr_matrix, similarity: csr_matrix
) -> np.ndarray:
    """累计用户历史物品与全部候选物品的相似度。"""
    user_history = train_matrix.getrow(user_index)
    scores = user_history @ similarity
    return np.asarray(scores.toarray()).ravel().astype(np.float32)


def stable_top_k(
    scores: np.ndarray,
    seen_indices: np.ndarray,
    item_ids: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """过滤已交互物品；同分按原始 item_id 升序。"""
    candidate_mask = np.ones(len(item_ids), dtype=bool)
    candidate_mask[seen_indices] = False
    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_scores = np.nan_to_num(scores[candidate_indices], nan=-np.inf)
    order = np.lexsort((item_ids[candidate_indices], -candidate_scores))
    selected = candidate_indices[order[: min(k, len(order))]]
    return [(int(item_ids[i]), float(scores[i])) for i in selected]
