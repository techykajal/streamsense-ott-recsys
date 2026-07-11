"""
ab_eval.py — offline A/B / experimentation on the TRAINED PyTorch ranker.

Loads the real trained model (artifacts/ranker.pt) and the real interactions, then
compares two variants by ranking each user's items and scoring against real labels:
  A = popularity baseline   (score = how often a movie is liked overall)
  B = trained PyTorch ranker (score = model's predicted P(engagement))
Metrics: Recall@K and NDCG@K, averaged over users.

NOTE (honesty): the scaffold's ranker trains on all interactions, so this is an
IN-SAMPLE comparison — it demonstrates the ranker orders items better than a popularity
baseline. For a leakage-free number, add a train/test split in ranking_torch.py and
evaluate only on the held-out rows.
"""
import json
import numpy as np
import pandas as pd
import torch
from ranking_torch import Ranker

K = 10
ART = "artifacts"


def ndcg_at_k(labels, k=K):
    labels = np.asarray(labels, dtype=float)
    top = labels[:k]
    gains = top / np.log2(np.arange(2, len(top) + 2))
    ideal = np.sort(labels)[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return gains.sum() / idcg if idcg > 0 else 0.0


def recall_at_k(labels, k=K):
    labels = np.asarray(labels, dtype=float)
    total = labels.sum()
    return labels[:k].sum() / total if total > 0 else 0.0


def load():
    df = pd.read_parquet("data/processed/interactions.parquet")
    maps = json.load(open(f"{ART}/id_maps.json"))
    ck = torch.load(f"{ART}/ranker.pt", map_location="cpu")
    model = Ranker(ck["n_user"], ck["n_movie"], ck["n_seg"], ck["dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return df, maps["uid"], maps["mid"], model


def ranker_scores(df, uid, mid, model):
    u = torch.tensor(df.user_id.map(uid).values, dtype=torch.long)
    m = torch.tensor(df.movie_id.map(mid).values, dtype=torch.long)
    s = torch.tensor(df.segment.values, dtype=torch.long)
    with torch.no_grad():
        return model(u, m, s).numpy()


def evaluate(df, score_col):
    ndcgs, recalls = [], []
    for _, g in df.groupby("user_id"):
        if g.label.sum() == 0 or len(g) < 2:   # need at least one positive to rank
            continue
        g = g.sort_values(score_col, ascending=False)
        ndcgs.append(ndcg_at_k(g.label.values))
        recalls.append(recall_at_k(g.label.values))
    return float(np.mean(ndcgs)), float(np.mean(recalls))


def main():
    df, uid, mid, model = load()

    # Variant A — popularity baseline: score = how many users liked the movie.
    pop = df.groupby("movie_id").label.sum()
    df["score_pop"] = df.movie_id.map(pop)

    # Variant B — the trained ranker's predicted engagement probability.
    df["score_ranker"] = ranker_scores(df, uid, mid, model)

    rows = []
    for name, col in [("A_popularity", "score_pop"), ("B_ranker", "score_ranker")]:
        ndcg, recall = evaluate(df, col)
        rows.append({"variant": name, f"NDCG@{K}": round(ndcg, 4),
                     f"Recall@{K}": round(recall, 4)})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    lift = (rows[1][f"NDCG@{K}"] / rows[0][f"NDCG@{K}"] - 1) * 100
    n_users = df.groupby("user_id").filter(lambda g: g.label.sum() > 0 and len(g) >= 2)\
                .user_id.nunique()
    print(f"\nEvaluated on {n_users} users (in-sample).")
    print(f"Ranker lift over popularity (NDCG@{K}): {lift:+.1f}%")


if __name__ == "__main__":
    main()