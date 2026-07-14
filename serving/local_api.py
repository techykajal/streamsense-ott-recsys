"""
local_api.py — run all three models locally (native, no Docker), exposing the SAME REST endpoints
the dashboard calls. This lets the whole demo run on an Apple Silicon Mac, where TensorFlow Serving
has no native build. TensorFlow and PyTorch both run natively on macOS, so we load the committed
SavedModels/checkpoint and mimic the TF Serving + TorchServe request/response shapes.

(The *real* TF Serving / TorchServe / Triton runtimes are demonstrated in notebooks/StreamSense_Colab_Demo.ipynb.
 This file is the convenient local stand-in so `streamlit run` "just works" on your laptop.)

Run:
    pip install fastapi uvicorn "tensorflow==2.15.1" torch pandas pyarrow
    python serving/local_api.py                    # serves on http://localhost:8000

Then point the dashboard at it (one port, three routes) and launch Streamlit:
    RETRIEVAL_URL=http://localhost:8000/v1/models/retrieval:predict \
    RANKER_URL=http://localhost:8000/v1/models/ranker:predict \
    TORCH_URL=http://localhost:8000/predictions/ranker \
    streamlit run app/streamsense_explorer.py
"""
import os, sys, glob, json, base64
from fastapi import FastAPI, Request
import uvicorn

sys.path.append("src")
import tensorflow as tf
import torch
from ranking_torch import Ranker

app = FastAPI(title="StreamSense local serving")

# ---- load the committed models once at startup ----
RET_DIR = "models/retrieval" if os.path.exists("models/retrieval") else "artifacts/retrieval"
retrieval = tf.saved_model.load(RET_DIR)

_tfx = sorted(glob.glob("models/ranking_tf/*/")) or sorted(glob.glob("artifacts/ranking_tf/*/"))
tfx_serve = tf.saved_model.load(_tfx[-1]).signatures["serving_default"] if _tfx else None

_ck = torch.load("models/ranker.pt" if os.path.exists("models/ranker.pt") else "artifacts/ranker.pt",
                 map_location="cpu")
torch_model = Ranker(_ck["n_user"], _ck["n_movie"], _ck["n_seg"], _ck["dim"])
torch_model.load_state_dict(_ck["state_dict"]); torch_model.eval()

print(f"loaded: retrieval={RET_DIR} | tfx_ranker={'yes' if tfx_serve else 'MISSING'} | torch_ranker=yes")


# ---- health endpoints (match TF Serving / TorchServe) ----
@app.get("/v1/models/retrieval")
def _r_health(): return {"model_version_status": [{"state": "AVAILABLE"}]}

@app.get("/v1/models/ranker")
def _k_health(): return {"model_version_status": [{"state": "AVAILABLE"}]}

@app.get("/ping")
def _ping(): return {"status": "Healthy"}


# ---- retrieval (two-tower) — TF Serving shape ----
@app.post("/v1/models/retrieval:predict")
async def retrieve(req: Request):
    inp = (await req.json())["inputs"]
    out = retrieval({"user_id": tf.constant([str(x) for x in inp["user_id"]]),
                     "segment": tf.constant([int(x) for x in inp["segment"]], tf.int64)})
    scores = out[0].numpy().tolist()
    ids = [[(x.decode() if isinstance(x, bytes) else str(x)) for x in row] for row in out[1].numpy()]
    return {"outputs": {"scores": scores, "ids": ids}}


# ---- TFX ranker — TF Serving shape (parses serialized tf.Example) ----
@app.post("/v1/models/ranker:predict")
async def rank_tfx(req: Request):
    if tfx_serve is None:
        return {"error": "models/ranking_tf not present"}
    body = await req.json()
    exs = [base64.b64decode(e["b64"]) for e in body["inputs"]["examples"]]
    pred = tfx_serve(examples=tf.constant(exs))
    arr = list(pred.values())[0].numpy().reshape(-1).tolist()
    return {"outputs": {"prediction": [[p] for p in arr]}}


# ---- PyTorch ranker — TorchServe shape (list of {user,movie,seg} indices) ----
@app.post("/predictions/ranker")
async def rank_torch(req: Request):
    rows = await req.json()
    if isinstance(rows, dict): rows = [rows]
    u = torch.tensor([int(r["user"]) for r in rows])
    m = torch.tensor([int(r["movie"]) for r in rows])
    s = torch.tensor([int(r["seg"]) for r in rows])
    with torch.no_grad():
        sc = torch_model(u, m, s).numpy().tolist()
    return [{"score": float(x)} for x in sc]


if __name__ == "__main__":
    print("StreamSense local serving → http://localhost:8000  (retrieval + TFX ranker + torch ranker)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
