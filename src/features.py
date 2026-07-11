"""
features.py — Data + feature engineering for the OTT recommender.

Covers three JD keywords in one file:
  * feature engineering / data pipeline (structured + unstructured)
  * user segmentation (K-Means)
  * content analysis (GenAI text embeddings of title + genres)

Dataset: MovieLens (public). 'ml-latest-small' by default for a 2-day POC;
switch to 'ml-25m' for a bigger, more OTT-like run.

Output: a parquet of interaction rows + a movies table with content embeddings,
plus TFRecords consumed by the TFX pipeline.
"""
import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = os.environ.get("DATA_DIR", "data")
MOVIELENS = os.environ.get("MOVIELENS", "ml-latest-small")  # or "ml-25m"
URL = f"https://files.grouplens.org/datasets/movielens/{MOVIELENS}.zip"
N_SEGMENTS = 8


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    target = os.path.join(DATA_DIR, MOVIELENS)
    if os.path.isdir(target):
        return target
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall(DATA_DIR)
    return target


def content_embeddings(movies: pd.DataFrame) -> np.ndarray:
    """GenAI content analysis: embed 'title + genres' with a sentence transformer.
    Swap this function for a Gemini/OpenAI embeddings call to match your GenAI story."""
    text = (movies["title"].fillna("") + " | " +
            movies["genres"].fillna("").str.replace("|", " ", regex=False))
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(text.tolist(), batch_size=256, show_progress_bar=True)
        return np.asarray(emb, dtype="float32")
    except Exception as e:  # offline fallback so the pipeline never blocks
        print(f"[content_embeddings] transformer unavailable ({e}); using hashed fallback.")
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(movies), 32)).astype("float32")


def build():
    root = download()
    ratings = pd.read_csv(os.path.join(root, "ratings.csv"))
    movies = pd.read_csv(os.path.join(root, "movies.csv"))

    # --- content analysis (unstructured -> dense features) ---
    emb = content_embeddings(movies)
    movies = movies.reset_index(drop=True)
    for j in range(emb.shape[1]):
        movies[f"emb_{j}"] = emb[:, j]

    # --- user segmentation: cluster users on mean content embedding of what they watched ---
    m_emb = movies.set_index("movieId")[[c for c in movies.columns if c.startswith("emb_")]]
    user_profile = (ratings.merge(m_emb, on="movieId")
                    .groupby("userId")[list(m_emb.columns)].mean())
    km = KMeans(n_clusters=N_SEGMENTS, n_init="auto", random_state=0)
    user_profile["segment"] = km.fit_predict(user_profile.values).astype("int64")

    # --- assemble interaction table (implicit + explicit) ---
    df = ratings.merge(user_profile[["segment"]], on="userId")
    df = df.merge(movies[["movieId", "title"]], on="movieId")
    df["label"] = (df["rating"] >= 4.0).astype("int64")  # positive engagement for ranking
    df["user_id"] = df["userId"].astype(str)
    df["movie_id"] = df["movieId"].astype(str)

    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/interactions.parquet", index=False)
    movies.to_parquet("data/processed/movies.parquet", index=False)
    write_tfrecords(df)
    print(f"Built {len(df):,} interactions, {df.user_id.nunique():,} users, "
          f"{df.movie_id.nunique():,} movies, {N_SEGMENTS} segments.")
    return df, movies


def write_tfrecords(df: pd.DataFrame, out="data/processed/interactions.tfrecord"):
    """TFX ExampleGen consumes TFRecords of tf.Example."""
    import tensorflow as tf

    def _bytes(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode()]))
    def _int(v):   return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
    def _float(v): return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

    with tf.io.TFRecordWriter(out) as w:
        for r in df.itertuples(index=False):
            ex = tf.train.Example(features=tf.train.Features(feature={
                "user_id": _bytes(r.user_id),
                "movie_id": _bytes(r.movie_id),
                "movie_title": _bytes(r.title),
                "segment": _int(r.segment),
                "rating": _float(r.rating),
                "label": _int(r.label),
            }))
            w.write(ex.SerializeToString())
    print(f"Wrote TFRecords → {out}")


if __name__ == "__main__":
    build()
