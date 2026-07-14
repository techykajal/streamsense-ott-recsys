"""
StreamSense Explorer — Hugging Face Spaces version (self-contained, models loaded in-process).

A Space runs a single process, so this build loads the committed models directly (native
TensorFlow + PyTorch) instead of calling a separate serving API. Same UI as the local dashboard:
  One Stage Ranker (TFX)  |  Two Stage (Retrieval: TF + Ranker: PyTorch)

Deploy: put this file as app.py at the Space root, add requirements.txt, and the models/ and
data/processed/ folders. See huggingface_space/README.md.
"""
import os, glob, json, base64
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="StreamSense Explorer", layout="wide")


# ----------------------------- data + models (cached) -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    inter = pd.read_parquet("data/processed/interactions.parquet")
    movies = pd.read_parquet("data/processed/movies.parquet")
    movies["movie_id"] = movies["movieId"].astype(str)
    id2title = dict(zip(movies.movie_id, movies.title))
    user_seg = inter.drop_duplicates("user_id").set_index("user_id")["segment"].to_dict()
    emb_cols = [c for c in movies.columns if c.startswith("emb_")]
    maps = json.load(open("models/id_maps.json"))
    return inter, movies, id2title, user_seg, emb_cols, maps


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    import tensorflow as tf
    import torch, torch.nn as nn

    class Ranker(nn.Module):
        def __init__(self, n_user, n_movie, n_seg, dim=32):
            super().__init__()
            self.u = nn.Embedding(n_user, dim); self.m = nn.Embedding(n_movie, dim)
            self.s = nn.Embedding(n_seg, dim)
            self.mlp = nn.Sequential(nn.Linear(dim*3, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, user, movie, seg):
            x = torch.cat([self.u(user), self.m(movie), self.s(seg)], dim=-1)
            return torch.sigmoid(self.mlp(x)).squeeze(-1)

    retrieval = tf.saved_model.load("models/retrieval")
    tfx_dirs = sorted(glob.glob("models/ranking_tf/*/"))
    tfx = tf.saved_model.load(tfx_dirs[-1]).signatures["serving_default"] if tfx_dirs else None
    ck = torch.load("models/ranker.pt", map_location="cpu")
    rk = Ranker(ck["n_user"], ck["n_movie"], ck["n_seg"], ck["dim"])
    rk.load_state_dict(ck["state_dict"]); rk.eval()
    return tf, torch, retrieval, tfx, rk


# ----------------------------- scoring (in-process) -----------------------------
def _serialize(tf, u, m, s, title=""):
    f = lambda **k: tf.train.Feature(**k)
    ex = tf.train.Example(features=tf.train.Features(feature={
        "user_id":     f(bytes_list=tf.train.BytesList(value=[str(u).encode()])),
        "movie_id":    f(bytes_list=tf.train.BytesList(value=[str(m).encode()])),
        "movie_title": f(bytes_list=tf.train.BytesList(value=[str(title).encode()])),
        "segment":     f(int64_list=tf.train.Int64List(value=[int(s)])),
        "rating":      f(float_list=tf.train.FloatList(value=[0.0]))}))
    return ex.SerializeToString()


def tfx_score(tf, tfx, triples):
    exs = tf.constant([_serialize(tf, *t) for t in triples])
    pred = tfx(examples=exs)
    return list(pred.values())[0].numpy().reshape(-1).tolist()


def retrieve(tf, retrieval, user_id, seg, k=60):
    out = retrieval({"user_id": tf.constant([str(user_id)]),
                     "segment": tf.constant([int(seg)], tf.int64)})
    ids = out[1].numpy()[0]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ids][:k]


def torch_score(torch, rk, maps, user_id, seg, movie_ids):
    uid, mid = maps["uid"], maps["mid"]
    kept, rows = [], []
    for m in movie_ids:
        if str(user_id) in uid and str(m) in mid:
            kept.append(m); rows.append((uid[str(user_id)], mid[str(m)], int(seg)))
    if not rows:
        return [], []
    u = torch.tensor([r[0] for r in rows]); mm = torch.tensor([r[1] for r in rows])
    ss = torch.tensor([r[2] for r in rows])
    with torch.no_grad():
        sc = rk(u, mm, ss).numpy().tolist()
    return kept, sc


def torch_score_users(torch, rk, maps, user_seg, users, mid):
    uid, mm = maps["uid"], maps["mid"]
    kept, rows = [], []
    for u in users:
        if str(u) in uid and str(mid) in mm:
            kept.append(u); rows.append((uid[str(u)], mm[str(mid)], int(user_seg.get(u, 0))))
    if not rows:
        return [], []
    uu = torch.tensor([r[0] for r in rows]); mmm = torch.tensor([r[1] for r in rows])
    ss = torch.tensor([r[2] for r in rows])
    with torch.no_grad():
        sc = rk(uu, mmm, ss).numpy().tolist()
    return kept, sc


# ----------------------------- UI -----------------------------
st.title("🎬 StreamSense Explorer")

inter, movies, id2title, user_seg, emb_cols, maps = load_data()
tf, torch, retrieval, tfx, rk = load_models()

with st.sidebar:
    st.header("Serving mode")
    mode = st.radio("Strategy", ["One Stage Ranker (TFX)",
                                 "Two Stage (Retrieval: TF + Ranker: PyTorch)"])
    two_stage = mode.startswith("Two Stage")
    if two_stage:
        st.caption("Two-tower retrieval shortlists candidates, then the PyTorch ranker orders them "
                   "— the pattern for millions of items.")
    else:
        st.caption("A single TFX ranker scores the catalogue directly — no retrieval. Fine because "
                   "this catalogue is small.")
    topk = st.slider("How many recommendations", 5, 20, 10)
    scan_cap = st.slider("One-stage: candidates to score", 500, 9800, 2000, step=500)

tab1, tab2, tab3 = st.tabs(["👤 For a user", "🎞️ For a movie", "📊 Model metrics"])


def recommend_for_user(uid, seg):
    if two_stage:
        cands = retrieve(tf, retrieval, uid, seg, k=60)
        ids, scores = torch_score(torch, rk, maps, uid, seg, cands)
        return pd.DataFrame({"movie_id": ids, "score": scores})
    seen = set(inter[inter.user_id == uid].movie_id)
    cands = [m for m in movies.movie_id if m not in seen][:scan_cap]
    scores = tfx_score(tf, tfx, [(uid, m, seg, id2title.get(m, "")) for m in cands])
    return pd.DataFrame({"movie_id": cands, "score": scores})


with tab1:
    st.caption(f"Mode: **{'Two Stage (Retrieval → PyTorch ranker)' if two_stage else 'One Stage Ranker (TFX)'}**")
    users = sorted(inter.user_id.unique(), key=lambda x: int(x))
    uid = st.selectbox("Select a user", users, key="u")
    seg = int(user_seg.get(uid, 0))
    st.write(f"**Segment:** {seg}")
    hist = inter[(inter.user_id == uid) & (inter.label == 1)].sort_values("rating", ascending=False)
    st.subheader("Recently liked (from history)")
    st.dataframe(hist[["title", "rating"]].head(8), hide_index=True, width="stretch")

    if st.button("Get recommendations", type="primary"):
        recs = recommend_for_user(uid, seg).assign(title=lambda d: d.movie_id.map(id2title)) \
                 .sort_values("score", ascending=False).head(topk)
        st.subheader("Top recommendations")
        st.bar_chart(recs.set_index("title")["score"])
        st.dataframe(recs[["title", "score"]], hide_index=True, width="stretch")

with tab2:
    st.caption("Content-similar titles use embeddings; likely-viewers use the current mode's ranker.")
    title = st.selectbox("Select a movie", sorted(movies.title.unique()), key="m")
    row = movies[movies.title == title].iloc[0]; mid = row.movie_id
    if emb_cols:
        V = movies[emb_cols].values; v = row[emb_cols].values.astype(float)
        sim = V @ v / (np.linalg.norm(V, axis=1) * np.linalg.norm(v) + 1e-9)
        near = movies.assign(similarity=sim)
        near = near[near.movie_id != mid].sort_values("similarity", ascending=False).head(10)
        st.subheader("Content-similar titles (embedding cosine)")
        st.dataframe(near[["title", "similarity"]], hide_index=True, width="stretch")
    if st.button("Find users likely to watch", type="primary"):
        users = sorted(inter.user_id.unique(), key=lambda x: int(x))
        if two_stage:
            r, s = torch_score_users(torch, rk, maps, user_seg, users, mid)
            top = pd.DataFrame({"user_id": r, "score": s})
        else:
            s = tfx_score(tf, tfx, [(u, mid, int(user_seg.get(u, 0)), title) for u in users])
            top = pd.DataFrame({"user_id": users, "score": s})
        top = top.sort_values("score", ascending=False).head(15)
        st.subheader(f"Users most likely to engage with “{title}”")
        st.bar_chart(top.set_index("user_id")["score"])
        st.dataframe(top, hide_index=True, width="stretch")

with tab3:
    st.subheader("📊 Per-model offline evaluation — real metrics on the real dataset")
    if os.path.exists("metrics/eval_metrics.json"):
        m = json.load(open("metrics/eval_metrics.json"))
        cards = st.columns(len(m) or 1)
        for col, (name, d) in zip(cards, m.items()):
            with col:
                hk = next((k for k in ("AUC", "NDCG@10", "HitRate@10") if k in d), list(d)[0])
                st.metric(name, f"{d[hk]:.3f}", help=f"headline: {hk}")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ranking quality — AUC**")
            auc = {n: d["AUC"] for n, d in m.items() if "AUC" in d}
            if auc: st.bar_chart(pd.DataFrame({"AUC": auc}))
        with c2:
            st.markdown("**Two-tower retrieval — Recall@K & hit-rate**")
            ret = {k: v for k, v in m.get("Two-tower retrieval", {}).items() if isinstance(v, (int, float))}
            if ret: st.bar_chart(pd.DataFrame({"score": ret}))
        tcols = st.columns(len(m) or 1)
        for col, (name, d) in zip(tcols, m.items()):
            with col:
                st.markdown(f"*{name}*")
                st.dataframe(pd.DataFrame({"metric": list(d), "value": [round(v, 4) if isinstance(v, (int, float)) else v for v in d.values()]}),
                             hide_index=True, width="stretch")
    else:
        st.info("metrics/eval_metrics.json not found.")
