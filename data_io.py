from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
EXPECTED_MD5 = "0e33842e24a9c977be4e0107933c0723"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_and_extract(data_dir: Path) -> Path:
    """下载、校验并只解压实验需要的两个文件。"""
    zip_path = data_dir / "ml-100k.zip"
    part_path = data_dir / "ml-100k.zip.part"
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "rec-lab/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        with part_path.open("wb") as target:
            shutil.copyfileobj(response, target)
    if _md5(part_path) != EXPECTED_MD5:
        part_path.unlink(missing_ok=True)
        raise ValueError("MovieLens ZIP 的 MD5 与官方值不一致")
    part_path.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise zipfile.BadZipFile("ZIP CRC 校验失败")
        members = ["ml-100k/u.data", "ml-100k/u.item"]
        if not set(members).issubset(archive.namelist()):
            raise FileNotFoundError("ZIP 中缺少 u.data 或 u.item")
        for member in members:
            archive.extract(member, data_dir)
    return data_dir / "ml-100k"


def _write_fallback(data_dir: Path) -> Path:
    """生成无网络也可运行的微型显式评分数据。"""
    fallback = data_dir / "fallback"
    fallback.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, int, int, int]] = []
    timestamp = 1_700_000_000
    for user_id in range(1, 19):
        group = (user_id - 1) % 3
        group_items = list(range(3 + group * 9, 12 + group * 9))
        offset = (user_id - 1) // 3
        chosen = [group_items[(offset + j) % 9] for j in range(6)]
        for item_id in [1, 2, *chosen]:
            rows.append((user_id, item_id, 5 if item_id != 2 else 4, timestamp))
            timestamp += 1
        negative = 3 + ((group + 1) % 3) * 9
        rows.append((user_id, negative, 2, timestamp))
        timestamp += 1
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    movies = pd.DataFrame({
        "item_id": range(1, 31),
        "title": [f"Offline Movie {item_id:02d}" for item_id in range(1, 31)],
    })
    ratings.to_csv(fallback / "ratings.csv", index=False)
    movies.to_csv(fallback / "movies.csv", index=False)
    return fallback


def acquire_dataset(data_dir: Path) -> tuple[Path, str]:
    """优先使用官方数据缓存；网络失败时返回内置微型数据。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "ml-100k"
    cached_files = [extracted / "u.data", extracted / "u.item"]
    if os.getenv("RECSYS_FORCE_FALLBACK") != "1" and all(
        path.exists() for path in cached_files
    ):
        return extracted, "movielens-100k-cache"
    if os.getenv("RECSYS_FORCE_FALLBACK") == "1":
        return _write_fallback(data_dir), "offline-fallback-forced"
    try:
        return _download_and_extract(data_dir), "movielens-100k"
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"[warning] 官方数据不可用，改用微型后备数据：{error}")
        return _write_fallback(data_dir), "offline-fallback"


def load_and_clean(root: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """读取评分和片名，并移除非法值与重复评分。"""
    if source.startswith("movielens"):
        ratings = pd.read_csv(
            root / "u.data", sep="\t", header=None,
            names=["user_id", "item_id", "rating", "timestamp"], dtype="int64",
        )
        movies = pd.read_csv(
            root / "u.item", sep="|", header=None, usecols=[0, 1],
            names=["item_id", "title"], encoding="latin-1",
        )
    else:
        ratings = pd.read_csv(root / "ratings.csv")
        movies = pd.read_csv(root / "movies.csv")
    raw_count = len(ratings)
    ratings = ratings.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    ratings = ratings[ratings["rating"].between(1, 5)].copy()
    ratings = ratings.sort_values(
        ["user_id", "item_id", "timestamp"], kind="mergesort"
    ).drop_duplicates(["user_id", "item_id"], keep="last")
    for column in ["user_id", "item_id", "rating", "timestamp"]:
        ratings[column] = ratings[column].astype("int64")
    movies = movies.drop_duplicates("item_id", keep="last").copy()
    movies["item_id"] = movies["item_id"].astype("int64")
    movies["title"] = movies["title"].astype(str)
    return ratings.reset_index(drop=True), movies, raw_count
