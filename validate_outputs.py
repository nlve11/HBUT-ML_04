from pathlib import Path

import pandas as pd

output_dir = Path("outputs")
train = pd.read_csv(output_dir / "train_positive.csv")
test = pd.read_csv(output_dir / "test_positive.csv")
recommendations = pd.read_csv(output_dir / "user_recommendations.csv")
summary = pd.read_csv(output_dir / "summary_metrics.csv")

split_overlap = train.merge(test, on=["user_id", "item_id"], how="inner")
assert split_overlap.empty, "训练集与测试集发生交互重叠"
seen_overlap = recommendations.merge(
    train[["user_id", "item_id"]], on=["user_id", "item_id"], how="inner"
)
assert seen_overlap.empty, "推荐结果仍包含已交互物品"
assert summary[["recall_at_10", "ndcg_at_10"]].apply(
    lambda series: series.between(0, 1).all()
).all()
assert not recommendations.duplicated(["method", "user_id", "rank"]).any()
assert recommendations["rank"].between(1, 10).all()
print(f"split_overlap={len(split_overlap)}")
print(f"seen_overlap={len(seen_overlap)}")
print("output_checks=passed")
