"""
StreamSense Explorer — hosted demo (lightweight reader).

The recommendations shown here were precomputed from the real trained models:
  • Two-tower retrieval (TensorFlow / TFRS)
  • PyTorch ranker (two-stage)
  • TFX/Keras ranker (one-stage)

To keep the free hosted app fast and within memory limits, this build reads those
precomputed results (no TensorFlow/PyTorch at runtime). Content-similar titles are
still computed live from the movie embeddings. The full training pipeline (TFX),
per-model evaluation, and the production serving runtimes (TF Serving / TorchServe /
NVIDIA Triton) live in the main repo.
"""
import os, json
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root (Streamlit runs main file from repo root)
PRE  = os.path.join(HERE, "precomputed")

def _p(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]

st.set_page_config(page_title="StreamSense Explorer", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    inter  = pd.read_parquet(_p(os.path.join(ROOT, "data/processed/interactions.parquet"),
                                os.path.join(HERE, "data/interactions.parquet")))
    movies = pd.read_parquet(_p(os.path.join(ROOT, "data/processed/movies.parquet"),
                                os.path.join(HERE, "data/movies.parquet")))
    movies["movie_id"] = movies["movieId"].astype(str)
    id2title = dict(zip(movies.movie_id, movies.title))
    user_seg = inter.drop_duplicates("user_id").set_index("user_id")["segment"].astype(int).to_dict()
    emb_cols = [c for c in movies.columns if c.startswith("emb_")]
    recs = json.load(open(os.path.join(PRE, "recs_by_user.json")))
    mu   = json.load(open(os.path.join(PRE, "users_by_movie.json")))
    return inter, movies, id2title, user_seg, emb_cols, recs, mu


inter, movies, id2title, user_seg, emb_cols, RECS, MU = load_data()

st.title("🎬 StreamSense Explorer")
st.caption("Recommendations precomputed from the real trained models "
           "(two-tower retrieval + PyTorch ranker + TFX ranker). "
           "Full training, evaluation, and live serving stack in the main repo.")

with st.sidebar:
    st.header("Serving mode")
    mode = st.radio("Strategy", ["One Stage Ranker (TFX)",
                                 "Two Stage (Retrieval: TF + Ranker: PyTorch)"])
    two_stage = mode.startswith("Two Stage")
    key = "two_stage" if two_stage else "one_stage"
    if two_stage:
        st.caption("Two-tower retrieval shortlists candidates, then the PyTorch ranker "
                   "orders them — the pattern for millions of items.")
    else:
        st.caption("A single TFX ranker scores the catalogue directly — no retrieval. "
                   "Fine because this catalogue is small.")
    topk = st.slider("How many recommendations", 5, 20, 10)

tab1, tab2, tab3 = st.tabs(["👤 For a user", "🎞️ For a movie", "📊 Model metrics"])

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
        pairs = RECS.get(key, {}).get(uid, [])[:topk]
        if not pairs:
            st.info("No precomputed recommendations for this user in this mode.")
        else:
            recs = pd.DataFrame(pairs, columns=["movie_id", "score"])
            recs["title"] = recs.movie_id.map(id2title)
            st.subheader("Top recommendations")
            st.bar_chart(recs.set_index("title")["score"])
            st.dataframe(recs[["title", "score"]], hide_index=True, width="stretch")

with tab2:
    st.caption("Content-similar titles use embeddings (computed live); "
               "likely-viewers use the current mode's ranker (precomputed).")
    title = st.selectbox("Select a movie", sorted(movies.title.unique()), key="m")
    row = movies[movies.title == title].iloc[0]; mid = row.movie_id
    if emb_cols:
        V = movies[emb_cols].values
        v = row[emb_cols].values.astype(float)
        sim = V @ v / (np.linalg.norm(V, axis=1) * np.linalg.norm(v) + 1e-9)
        near = movies.assign(similarity=sim)
        near = near[near.movie_id != mid].sort_values("similarity", ascending=False).head(10)
        st.subheader("Content-similar titles (embedding cosine)")
        st.dataframe(near[["title", "similarity"]], hide_index=True, width="stretch")
    if st.button("Find users likely to watch", type="primary"):
        pairs = MU.get(key, {}).get(mid, [])[:15]
        if not pairs:
            st.info("No precomputed viewers for this movie in this mode.")
        else:
            top = pd.DataFrame(pairs, columns=["user_id", "score"])
            st.subheader(f"Users most likely to engage with “{title}”")
            st.bar_chart(top.set_index("user_id")["score"])
            st.dataframe(top, hide_index=True, width="stretch")

with tab3:
    st.subheader("📊 Per-model offline evaluation — real metrics on the real dataset")
    mp = _p(os.path.join(ROOT, "metrics/eval_metrics.json"),
            os.path.join(HERE, "metrics/eval_metrics.json"))
    if os.path.exists(mp):
        m = json.load(open(mp))
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
                st.dataframe(pd.DataFrame({"metric": list(d),
                    "value": [round(v, 4) if isinstance(v, (int, float)) else v for v in d.values()]}),
                    hide_index=True, width="stretch")
    else:
        st.info("metrics/eval_metrics.json not found.")
