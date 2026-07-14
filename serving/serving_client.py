"""
serving_client.py — thin client over the two TF Serving endpoints.

    RETRIEVAL  (two-tower)  -> http://localhost:8501/v1/models/retrieval:predict
    RANKER     (TFX/Keras)  -> http://localhost:8502/v1/models/ranker:predict

This is the online, two-stage inference path, wired through the serving APIs:
    user  --retrieval-->  top-K candidate movie_ids  --ranker-->  P(engagement) per candidate  -> sorted list

Used by the smoke test below and by the Streamlit dashboard (app/streamsense_explorer.py).
Requires TensorFlow only to serialize tf.Example for the ranker request.
"""
import os, json, base64
import requests

from make_ranker_request import request_json  # same folder

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8501/v1/models/retrieval:predict")
RANKER_URL    = os.environ.get("RANKER_URL",    "http://localhost:8502/v1/models/ranker:predict")


def retrieve(user_id, segment, k=50, timeout=10):
    """Call the retrieval model -> list of candidate movie_id strings (best first)."""
    payload = {"inputs": {"user_id": [str(user_id)], "segment": [int(segment)]}}
    r = requests.post(RETRIEVAL_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    outputs = r.json()["outputs"]
    # BruteForce returns (scores, identifiers); output names vary, so detect by dtype.
    ids, scores = None, None
    for _, val in (outputs.items() if isinstance(outputs, dict) else [("o", outputs)]):
        flat = val[0] if isinstance(val[0], list) else val
        if isinstance(flat[0], str):
            ids = [str(x) for x in flat]
        else:
            scores = [float(x) for x in flat]
    if ids is None:                       # single-output fallback
        ids = [str(x) for x in (val[0] if isinstance(val[0], list) else val)]
    return ids[:k]


def rank(user_id, segment, movie_ids, titles=None, timeout=15):
    """Call the ranker on (user, candidate) pairs -> list of (movie_id, score) sorted desc."""
    titles = titles or {}
    triples = [(user_id, mid, segment, titles.get(mid, "")) for mid in movie_ids]
    payload = request_json(triples)
    r = requests.post(RANKER_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json()["outputs"]
    preds = out["prediction"] if isinstance(out, dict) else out
    scores = [float(p[0]) if isinstance(p, list) else float(p) for p in preds]
    ranked = sorted(zip(movie_ids, scores), key=lambda z: z[1], reverse=True)
    return ranked


def recommend(user_id, segment, k_retrieve=50, k_final=10, titles=None):
    """Full two-stage recommendation for one user, entirely through the serving APIs."""
    candidates = retrieve(user_id, segment, k=k_retrieve)
    ranked = rank(user_id, segment, candidates, titles=titles)
    return ranked[:k_final]


if __name__ == "__main__":
    # Smoke test — run after both TF Serving containers are up.
    print("retrieval health:", requests.get(RETRIEVAL_URL.replace(":predict", "")).json())
    print("ranker health   :", requests.get(RANKER_URL.replace(":predict", "")).json())
    recs = recommend(user_id="1", segment=0, k_retrieve=50, k_final=10)
    print("\nTop-10 recommendations for user 1:")
    for mid, score in recs:
        print(f"  movie {mid:>6}  score={score:.4f}")
