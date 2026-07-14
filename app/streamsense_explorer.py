"""
StreamSense Explorer — interactive dashboard over the served models.

It talks to the serving endpoints (no in-process models), and lets you switch between two
serving strategies to show the scale trade-off:

  TESTING  — single TFX ranker, scores the catalogue directly (no retrieval).
             Fine here because the catalogue is small.  Endpoint:
                 Ranker (TFX)   http://localhost:8502/v1/models/ranker:predict   (tf_serving_tfx_run.sh)

  PROD     — two-stage: two-tower retrieval shortlists, then the PyTorch ranker orders.
             The pattern you need at millions-of-items scale.  Endpoints:
                 Retrieval      http://localhost:8501/v1/models/retrieval:predict (tf_serving_run.sh)
                 Ranker (Torch) http://localhost:8080/predictions/ranker          (torchserve_run.sh)

Run:  pip install -r app/requirements.txt ; streamlit run app/streamsense_explorer.py
(build data first: python src/features.py ; and start the serving processes you need per mode)
"""
import os, json, base64
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="StreamSense Explorer", layout="wide")

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8501/v1/models/retrieval:predict")
TFX_URL       = os.environ.get("RANKER_URL",    "http://localhost:8502/v1/models/ranker:predict")
TORCH_URL     = os.environ.get("TORCH_URL",      "http://localhost:8080/predictions/ranker")
DATA = "data/processed"


# ----------------------------- data -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    inter = pd.read_parquet(f"{DATA}/interactions.parquet")
    movies = pd.read_parquet(f"{DATA}/movies.parquet")
    movies["movie_id"] = movies["movieId"].astype(str)
    id2title = dict(zip(movies.movie_id, movies.title))
    user_seg = inter.drop_duplicates("user_id").set_index("user_id")["segment"].to_dict()
    emb_cols = [c for c in movies.columns if c.startswith("emb_")]
    maps_path = "models/id_maps.json" if os.path.exists("models/id_maps.json") else "artifacts/id_maps.json"
    maps = json.load(open(maps_path)) if os.path.exists(maps_path) else {"uid": {}, "mid": {}}
    return inter, movies, id2title, user_seg, emb_cols, maps


# --------------------- serving calls ----------------------------
def _example_b64(user_id, movie_id, segment, title=""):
    import tensorflow as tf   # lazy: metrics/UI work even without TF installed
    def _b(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(v).encode()]))
    def _i(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
    def _f(v): return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))
    ex = tf.train.Example(features=tf.train.Features(feature={
        "user_id": _b(user_id), "movie_id": _b(movie_id), "movie_title": _b(title),
        "segment": _i(segment), "rating": _f(0.0)}))
    return base64.b64encode(ex.SerializeToString()).decode()


def tfx_score(triples, timeout=30):
    """TFX ranker (TF Serving). triples: (user_id, movie_id, segment, title) -> scores in order."""
    payload = {"signature_name": "serving_default",
               "inputs": {"examples": [{"b64": _example_b64(*t)} for t in triples]}}
    out = requests.post(TFX_URL, json=payload, timeout=timeout).json()["outputs"]
    preds = out["prediction"] if isinstance(out, dict) else out
    return [float(p[0]) if isinstance(p, list) else float(p) for p in preds]


def _parse_torch(resp):
    """PyTorch serving returns [{'score':..}, ...] (or nested one level). -> list of floats."""
    flat = resp[0] if (isinstance(resp, list) and resp and isinstance(resp[0], list)) else resp
    out = []
    for d in flat if isinstance(flat, list) else []:
        if isinstance(d, dict) and "score" in d:
            out.append(float(d["score"]))
        elif isinstance(d, (int, float)):
            out.append(float(d))
    return out


def torch_score(user_id, segment, movie_ids, maps, timeout=30):
    """PyTorch ranker. Uses id_maps to convert ids->indices. Returns (ids, scores)."""
    uid, mid = maps["uid"], maps["mid"]
    rows, kept = [], []
    for m in movie_ids:
        if str(user_id) in uid and str(m) in mid:
            rows.append({"user": uid[str(user_id)], "movie": mid[str(m)], "seg": int(segment)})
            kept.append(m)
    if not rows:
        return [], []
    scores = _parse_torch(requests.post(TORCH_URL, json=rows, timeout=timeout).json())
    n = min(len(kept), len(scores))
    return kept[:n], scores[:n]


def retrieve(user_id, segment, k=60, timeout=15):
    payload = {"inputs": {"user_id": [str(user_id)], "segment": [int(segment)]}}
    outputs = requests.post(RETRIEVAL_URL, json=payload, timeout=timeout).json()["outputs"]
    ids = None
    items = outputs.items() if isinstance(outputs, dict) else [("o", outputs)]
    for _, val in items:
        flat = val[0] if isinstance(val[0], list) else val
        if isinstance(flat[0], str):
            ids = [str(x) for x in flat]
    if ids is None:
        v = list(outputs.values())[0] if isinstance(outputs, dict) else outputs
        ids = [str(x) for x in (v[0] if isinstance(v[0], list) else v)]
    return ids[:k]


def health(url):
    try:
        base = url.split("/v1/")[0] if "/v1/" in url else url.rsplit("/", 2)[0] + "/ping"
        requests.get(url.replace(":predict", "") if "/v1/" in url else base, timeout=4)
        return True
    except Exception:
        return False


# ----------------------------- UI -------------------------------
st.title("🎬 StreamSense Explorer")

try:
    inter, movies, id2title, user_seg, emb_cols, maps = load_data()
except Exception as e:
    st.error(f"Could not load data/processed/*.parquet. Run `python src/features.py` first. ({e})")
    st.stop()

with st.sidebar:
    st.header("Serving mode")
    mode = st.radio("Strategy", ["One Stage Ranker (TFX)",
                                 "Two Stage (Retrieval: TF + Ranker: PyTorch)"])
    is_prod = mode.startswith("Two Stage")
    if is_prod:
        st.caption("Two-tower retrieval shortlists candidates, then the PyTorch ranker orders them — "
                   "the pattern for millions of items. Needs retrieval (:8501) + PyTorch ranker (:8080).")
        st.write("Retrieval:", "🟢" if health(RETRIEVAL_URL) else "🔴")
        st.write("PyTorch ranker:", "🟢" if health(TORCH_URL) else "🔴")
    else:
        st.caption("A single TFX ranker scores the catalogue directly — no retrieval. Fine because "
                   "this catalogue is small. Needs the TFX ranker (:8502).")
        st.write("TFX ranker:", "🟢" if health(TFX_URL) else "🔴")
    topk = st.slider("How many recommendations", 5, 20, 10)
    scan_cap = st.slider("Testing: candidates to score", 500, 9800, 2000, step=500)

tab1, tab2, tab3 = st.tabs(["👤 For a user", "🎞️ For a movie", "📊 Model metrics"])


def recommend_for_user(uid, seg):
    if is_prod:
        # two-stage: retrieval shortlists candidates, PyTorch ranker orders them
        cands = retrieve(uid, seg, k=60)
        ids, scores = torch_score(uid, seg, cands, maps)
        return pd.DataFrame({"movie_id": ids, "score": scores})
    # one-stage: TFX ranker scores the unseen catalogue directly
    seen = set(inter[inter.user_id == uid].movie_id)
    cands = [m for m in movies.movie_id if m not in seen][:scan_cap]
    scores = tfx_score([(uid, m, seg, id2title.get(m, "")) for m in cands])
    return pd.DataFrame({"movie_id": cands, "score": scores})


def score_users_for_movie(mid, title):
    users = sorted(inter.user_id.unique(), key=lambda x: int(x))
    if is_prod:
        # two-stage mode uses the PyTorch ranker over all users
        r, s = torch_score_multi_user(users, mid, title)
        return pd.DataFrame({"user_id": r, "score": s})
    scores = tfx_score([(u, mid, int(user_seg.get(u, 0)), title) for u in users])
    return pd.DataFrame({"user_id": users, "score": scores})


def torch_score_multi_user(users, mid, title):
    uid, mm = maps["uid"], maps["mid"]
    rows, kept = [], []
    for u in users:
        if str(u) in uid and str(mid) in mm:
            rows.append({"user": uid[str(u)], "movie": mm[str(mid)], "seg": int(user_seg.get(u, 0))})
            kept.append(u)
    if not rows:
        return [], []
    scores = _parse_torch(requests.post(TORCH_URL, json=rows, timeout=30).json())
    n = min(len(kept), len(scores))
    return kept[:n], scores[:n]


# ---- Tab 1 ----
with tab1:
    st.caption(f"Mode: **{'Two Stage (Retrieval TF → PyTorch ranker)' if is_prod else 'One Stage Ranker (TFX)'}**")
    users = sorted(inter.user_id.unique(), key=lambda x: int(x))
    uid = st.selectbox("Select a user", users, key="u")
    seg = int(user_seg.get(uid, 0))
    st.write(f"**Segment:** {seg}")
    hist = inter[(inter.user_id == uid) & (inter.label == 1)].sort_values("rating", ascending=False)
    st.subheader("Recently liked (from history)")
    st.dataframe(hist[["title", "rating"]].head(8), hide_index=True, use_container_width=True)

    if st.button("Get recommendations", type="primary"):
        try:
            recs = recommend_for_user(uid, seg)
            recs = recs.assign(title=lambda d: d.movie_id.map(id2title)) \
                       .sort_values("score", ascending=False).head(topk)
            st.subheader("Top recommendations")
            st.bar_chart(recs.set_index("title")["score"])
            st.dataframe(recs[["title", "score"]], hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Serving call failed — is the right endpoint up for this mode? ({e})")

# ---- Tab 2 ----
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
        st.dataframe(near[["title", "similarity"]], hide_index=True, use_container_width=True)
    if st.button("Find users likely to watch", type="primary"):
        try:
            top = score_users_for_movie(mid, title).sort_values("score", ascending=False).head(15)
            st.subheader(f"Users most likely to engage with “{title}”")
            st.bar_chart(top.set_index("user_id")["score"])
            st.dataframe(top, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Ranker call failed — is the ranker up for this mode? ({e})")

# ---- Tab 3 ----
with tab3:
    st.subheader("📊 Per-model offline evaluation — real metrics on the real dataset")
    mp = "metrics/eval_metrics.json"
    if not os.path.exists(mp):
        st.info("Run notebooks/StreamSense_Evaluation.ipynb to generate metrics/eval_metrics.json.")
    else:
        m = json.load(open(mp))

        # headline cards
        cards = st.columns(len(m) or 1)
        head_keys = ("AUC", "NDCG@10", "HitRate@10")
        for col, (name, d) in zip(cards, m.items()):
            with col:
                hk = next((k for k in head_keys if k in d), list(d.keys())[0])
                st.metric(name, f"{d[hk]:.3f}", help=f"headline: {hk}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ranking quality — AUC (higher is better)**")
            auc = {name: d["AUC"] for name, d in m.items() if "AUC" in d}
            if auc:
                st.bar_chart(pd.DataFrame({"AUC": auc}))
        with c2:
            st.markdown("**Two-tower retrieval — Recall@K & hit-rate**")
            ret = {k: v for k, v in m.get("Two-tower retrieval", {}).items()
                   if isinstance(v, (int, float))}
            if ret:
                st.bar_chart(pd.DataFrame({"score": ret}))

        st.markdown("**Full metrics per model**")
        tcols = st.columns(len(m) or 1)
        for col, (name, d) in zip(tcols, m.items()):
            with col:
                st.markdown(f"*{name}*")
                st.dataframe(pd.DataFrame({"metric": list(d.keys()),
                                           "value": [round(v, 4) if isinstance(v, (int, float)) else v
                                                     for v in d.values()]}),
                             hide_index=True, use_container_width=True)
        st.caption("Generated by notebooks/StreamSense_Evaluation.ipynb — in-sample on MovieLens "
                   "(models trained on the full dataset).")
