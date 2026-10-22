import numpy as np
from scipy.sparse import csr_matrix

from src.models import (
    item_cf_scores,
    item_cosine_similarity,
    popularity_scores,
    stable_top_k,
)

matrix = csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.float32))
item_ids = np.array([10, 11, 12, 13])
popularity = popularity_scores(matrix)
similarity = item_cosine_similarity(matrix)
scores = item_cf_scores(0, matrix, similarity)
ranked = stable_top_k(scores, matrix.getrow(0).indices, item_ids, k=2)

assert popularity.tolist() == [2.0, 1.0, 1.0, 0.0]
assert np.allclose(similarity.diagonal(), 0.0)
assert [item_id for item_id, _ in ranked] == [12, 13]
print(f"popularity={popularity.tolist()}")
print(f"similarity_nnz={similarity.nnz}")
print(f"top2={[item_id for item_id, _ in ranked]}")
print("model_checks=passed")
