import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Data Loading
# =========================

def load_ratings(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        df = pd.read_csv(path)
        rename_map = {}
        if "userId" in df.columns:
            rename_map["userId"] = "user_id"
        if "movieId" in df.columns:
            rename_map["movieId"] = "item_id"
        if rename_map:
            df = df.rename(columns=rename_map)

        if "timestamp" not in df.columns:
            df["timestamp"] = 0

        return df[["user_id", "item_id", "rating", "timestamp"]]

    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="utf-8",
    )


def load_movies(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        df = pd.read_csv(path)
        rename_map = {}
        if "movieId" in df.columns:
            rename_map["movieId"] = "item_id"
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    columns = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "action", "adventure", "animation", "children", "comedy",
        "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
        "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western",
    ]
    return pd.read_csv(path, sep="|", names=columns, encoding="latin-1")


# =========================
# Data Processing
# =========================

def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        aggfunc="mean",
    )


def fill_missing_values(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.fillna(0)


def compute_sparsity(ratings: pd.DataFrame) -> float:
    num_users = ratings["user_id"].nunique()
    num_items = ratings["item_id"].nunique()
    total_possible = num_users * num_items

    if total_possible == 0:
        return 0.0

    return 1.0 - (len(ratings) / total_possible)


# =========================
# Similarity
# =========================

def compute_user_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(matrix.values)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


# =========================
# Recommendation
# =========================

def recommend_user_based(user_id: int, user_item_matrix: pd.DataFrame, similarity_matrix: pd.DataFrame, movies_df: pd.DataFrame,
    top_n: int = 10, top_k_similar: int = 20, min_similarity: float = 0.0,) -> pd.DataFrame:

    if user_id not in user_item_matrix.index:
        raise ValueError("User not found")

    user_ratings = user_item_matrix.loc[user_id]

    unseen_items = user_ratings[user_ratings == 0].index

    similarities = similarity_matrix.loc[user_id].drop(user_id)
    similarities = similarities[similarities > min_similarity]

    if similarities.empty:
        return pd.DataFrame(columns=["item_id", "score", "title"])

    similarities = similarities.sort_values(ascending=False).head(top_k_similar)

    similar_users_ratings = user_item_matrix.loc[similarities.index, unseen_items]

    sim_values = similarities.values.reshape(-1, 1)

    weighted_sum = (similar_users_ratings.values * sim_values).sum(axis=0)

    mask = (similar_users_ratings.values > 0).astype(float)
    sim_sum = (mask * sim_values).sum(axis=0)

    scores = np.divide(
        weighted_sum,
        sim_sum,
        out=np.zeros_like(weighted_sum),
        where=sim_sum != 0
    )

    recommendations = pd.DataFrame({
        "item_id": unseen_items,
        "score": scores
    })

    recommendations = recommendations.dropna()
    recommendations = recommendations.sort_values("score", ascending=False).head(top_n)

    recommendations = recommendations.merge(
        movies_df[["item_id", "title"]],
        on="item_id",
        how="left"
    )

    return recommendations.reset_index(drop=True)
