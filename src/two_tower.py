"""
two_tower.py — Retrieval (candidate generation) with TensorFlow Recommenders.

Query tower  : user_id + segment  -> user embedding
Candidate    : movie_id + title   -> item embedding
Trained with retrieval loss; evaluated with FactorizedTopK (recall@k).
Exports a SavedModel that returns top-K movie ids for a user -> served by TF Serving.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

EMB = 32
ARTIFACTS = os.environ.get("ARTIFACTS", "artifacts/retrieval")


def load():
    df = pd.read_parquet("data/processed/interactions.parquet")
    pos = df[df.label == 1]  # implicit positives for retrieval
    ds = tf.data.Dataset.from_tensor_slices({
        "user_id": pos.user_id.values,
        "segment": pos.segment.values.astype("int64"),
        "movie_id": pos.movie_id.values,
        "movie_title": pos.title.values,
    }).shuffle(100_000, seed=0)
    movies = tf.data.Dataset.from_tensor_slices(df.movie_id.unique())
    return ds, movies, df


class UserModel(tf.keras.Model):
    def __init__(self, user_ids, n_seg):
        super().__init__()
        self.u = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
        self.u_emb = tf.keras.layers.Embedding(len(user_ids) + 1, EMB)
        self.s_emb = tf.keras.layers.Embedding(n_seg + 1, EMB)

    def call(self, x):
        return self.u_emb(self.u(x["user_id"])) + self.s_emb(x["segment"])


class MovieModel(tf.keras.Model):
    def __init__(self, movie_ids):
        super().__init__()
        self.m = tf.keras.layers.StringLookup(vocabulary=movie_ids, mask_token=None)
        self.m_emb = tf.keras.layers.Embedding(len(movie_ids) + 1, EMB)

    def call(self, mid):
        return self.m_emb(self.m(mid))


class TwoTower(tfrs.Model):
    def __init__(self, user_ids, movie_ids, n_seg, candidates):
        super().__init__()
        self.user_model = UserModel(user_ids, n_seg)
        self.movie_model = MovieModel(movie_ids)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates.batch(512).map(self.movie_model)))

    def compute_loss(self, features, training=False):
        return self.task(self.user_model(features),
                         self.movie_model(features["movie_title"]) if False
                         else self.movie_model(features["movie_id"]))


def main():
    ds, movies, df = load()
    user_ids = df.user_id.unique()
    movie_ids = df.movie_id.unique()
    n_seg = int(df.segment.max())

    n_pos = int((df.label == 1).sum())          # number of positive interactions
    train_size = max(int(n_pos * 0.8), 1)        # 80/20 split — works for any dataset size
    train = ds.take(train_size).batch(2048).cache()
    test = ds.skip(train_size).batch(2048).cache()

    model = TwoTower(user_ids, movie_ids, n_seg, movies)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    model.fit(train, validation_data=test, epochs=3)

    # BruteForce index for serving: user features -> top-K movie ids
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=50)
    index.index_from_dataset(
        movies.batch(256).map(lambda mid: (mid, model.movie_model(mid))))
    # warm-up call so the SavedModel signature is traced
    _ = index({"user_id": tf.constant(["1"]), "segment": tf.constant([0], tf.int64)})

    os.makedirs(ARTIFACTS, exist_ok=True)
    tf.saved_model.save(index, ARTIFACTS)
    print(f"Saved retrieval index → {ARTIFACTS}")


if __name__ == "__main__":
    main()
