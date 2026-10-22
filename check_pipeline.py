import numpy as np
import pandas as pd

from src.pipeline import (
    build_interaction_matrix,
    explicit_to_positive,
    temporal_leave_one_out,
)

ratings = pd.DataFrame([
    (1, 10, 5, 1), (1, 11, 4, 2), (1, 12, 2, 3),
    (2, 10, 5, 1), (2, 13, 5, 2), (3, 14, 5, 1),
], columns=["user_id", "item_id", "rating", "timestamp"])
positives = explicit_to_positive(ratings, threshold=4)
train, test = temporal_leave_one_out(positives)
users = np.sort(test["user_id"].unique())
items = np.sort(positives["item_id"].unique())
matrix, _, _ = build_interaction_matrix(train, users, items)

assert len(positives) == 5
assert test.set_index("user_id")["item_id"].to_dict() == {1: 11, 2: 13}
assert matrix.shape == (2, 4) and matrix.nnz == 2
print("positives=5 train=2 test=2")
print("targets={1: 11, 2: 13}")
print("matrix_shape=(2, 4) nnz=2")
print("pipeline_checks=passed")
