import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# Data Loading
# =========================

def load_movies(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_credits(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_datasets(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    if "movie_id" in credits.columns:
        credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id")
    if "title" not in df.columns:
        if "title_x" in df.columns:
            df["title"] = df["title_x"]
        elif "title_y" in df.columns:
            df["title"] = df["title_y"]
    drop_cols = [c for c in ["title_x", "title_y"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


# =========================
# Feature Engineering
# =========================

def _parse_name_list(value: str, top_n: int | None = None) -> list[str]:
    if pd.isna(value):
        return []
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    names = [n for n in names if n]
    if top_n is not None:
        return names[:top_n]
    return names


def _get_director(value: str) -> list[str]:
    if pd.isna(value):
        return []
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            name = item.get("name", "")
            return [name] if name else []
    return []


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["genres_list"] = df["genres"].apply(_parse_name_list)
    df["keywords_list"] = df["keywords"].apply(_parse_name_list)
    df["cast_list"] = df["cast"].apply(lambda x: _parse_name_list(x, top_n=3))
    df["director_list"] = df["crew"].apply(_get_director)

    df["overview"] = df["overview"].fillna("")

    df["features"] = (
        df["overview"].astype(str)
        + " " + df["genres_list"].apply(lambda x: " ".join(x))
        + " " + df["keywords_list"].apply(lambda x: " ".join(x))
        + " " + df["cast_list"].apply(lambda x: " ".join(x))
        + " " + df["director_list"].apply(lambda x: " ".join(x))
    )

    return df


# =========================
# Vectorization
# =========================

def vectorize_features(df: pd.DataFrame, feature_col: str = "features"):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df[feature_col])
    return matrix, vectorizer


# =========================
# Similarity
# =========================

def compute_item_similarity(matrix) -> np.ndarray:
    return cosine_similarity(matrix)


# =========================
# Recommendation
# =========================

def recommend_content_based( title: str, df: pd.DataFrame,
    similarity_matrix: np.ndarray, top_n: int = 10,) -> pd.DataFrame:
    
    if "title" not in df.columns:
        raise ValueError("title column not found in dataset")

    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        raise ValueError("Movie title not found")

    idx = matches.index[0]
    scores = similarity_matrix[idx]

    ranked = (
        pd.DataFrame({"index": df.index, "score": scores})
        .sort_values("score", ascending=False)
    )

    ranked = ranked[ranked["index"] != idx].head(top_n)

    recommendations = df.loc[ranked["index"], ["id", "title"]].copy()
    recommendations["score"] = ranked["score"].values

    return recommendations.reset_index(drop=True)
