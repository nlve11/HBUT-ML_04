from src.evaluation import ndcg_at_k, recall_at_k

recommended = [9, 7, 5, 3]
relevant = {7}
recall = recall_at_k(recommended, relevant, k=3)
ndcg = ndcg_at_k(recommended, relevant, k=3)

assert recall == 1.0
assert abs(ndcg - 0.6309297536) < 1e-9
assert recall_at_k(recommended, set(), k=3) == 0.0
assert ndcg_at_k(recommended, set(), k=3) == 0.0
print(f"Recall@3={recall:.6f}")
print(f"NDCG@3={ndcg:.6f}")
print("metric_checks=passed")
