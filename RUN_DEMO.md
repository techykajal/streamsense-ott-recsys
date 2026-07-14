# Running the dashboard — hosted link or local live serving

There are two ways to see the **StreamSense Explorer** dashboard.

- **Just want to look?** Open the public, always-on version — no setup:
  **https://streamsense-ott-recsys-hhqytkrdkbhbs2kvtoufdc.streamlit.app/**
  It reads recommendations **precomputed from the real trained models** (retrieval + both rankers),
  so it needs no serving server and stays within the free tier. Content-similar titles are computed
  live from the embeddings.
- **Want to show the real serving path?** Run it locally (below). The local dashboard issues **live
  requests** to the serving runtimes — it never loads a model in-process.

---

## Local run — the real serving path

The dashboard offers two selectable strategies so you can demonstrate the scale trade-off:

```
 One Stage Ranker (TFX)   (small catalogue — one ranker is enough)
   user ─────────────────────────────▶ TFX ranker (TF Serving :8502) ─▶ scores ─▶ top-K

 Two Stage (Retrieval: TF + Ranker: PyTorch)   (millions-of-items pattern)
   user ─▶ Retrieval (TF Serving :8501) ─▶ top-K candidates
                                              └▶ PyTorch ranker (TorchServe :8080) ─▶ scores ─▶ top-K
```

| Endpoint | Model | Used by | Script |
|---|---|---|---|
| `:8501` | two-tower retrieval | Two Stage | `serving/tf_serving_run.sh` |
| `:8502` | TFX/Keras ranker | One Stage | `serving/tf_serving_tfx_run.sh` |
| `:8080` | PyTorch ranker | Two Stage | `serving/torchserve_run.sh` |

> **On Apple Silicon**, native TF Serving has no arm64 build and amd64 emulation is unreliable, so
> the repo also ships `serving/local_api.py` — a FastAPI stand-in that serves all three models in one
> process and mimics the TF Serving / TorchServe request shapes. `serving/run_local.sh` starts it
> together with the dashboard. Use this if the containers won't start locally.

## Prerequisites

The registered models ship in `models/`, so you don't need to retrain. If you want to rebuild from
scratch:

```bash
python src/features.py        # data/processed/*.parquet + interactions.tfrecord
python src/two_tower.py        # retrieval SavedModel  -> artifacts/retrieval
python src/ranking_torch.py    # PyTorch ranker        -> artifacts/ranker.pt
# TFX ranker: run notebooks/StreamSense_Colab_TFX.ipynb -> models/ranking_tf/<version>/
```

Docker running (for TF Serving), or use the Colab flow below.

## Step 1 — Start the endpoints you need

For **One Stage** mode you only need the TFX ranker. For **Two Stage** mode you need retrieval +
TorchServe. Start all three to switch freely in the dashboard:

```bash
bash serving/tf_serving_tfx_run.sh    # TFX ranker      -> :8502   (One Stage)
bash serving/tf_serving_run.sh        # two-tower       -> :8501   (Two Stage)
bash serving/torchserve_run.sh        # PyTorch ranker  -> :8080   (Two Stage)
```

Smoke-test the two-stage path through the APIs:

```bash
python serving/serving_client.py      # prints top-10 recommendations for user 1
```

## Step 2 — Run the dashboard

```bash
pip install -r app/requirements.txt
streamlit run app/streamsense_explorer.py
```

Open the URL Streamlit prints. Tabs:
- **For a user** — pick a user → segment + watch history + top-N recommendations (retrieval → ranker), with scores.
- **For a movie** — pick a title → content-similar titles + users most likely to watch it.
- **Model metrics** — real per-model metrics from `metrics/eval_metrics.json`.

## Step 3 — Populate the metrics tab

```bash
# run notebooks/StreamSense_Evaluation.ipynb  -> writes metrics/eval_metrics.json
```

Reload the dashboard's *Model metrics* tab to see the numbers.

---

## Running the whole demo inside Google Colab

TF Serving and Streamlit can both run in one Colab runtime; expose the dashboard with a tunnel.

```python
# after cloning the repo and starting the serving scripts:
!pip install -q streamlit
!npm install -g localtunnel
import subprocess, time
subprocess.Popen(["streamlit","run","app/streamsense_explorer.py","--server.port","8500"])
time.sleep(6)
!npx localtunnel --port 8500       # prints a public https URL for the dashboard
```

---

## Permanent free hosting (what powers the public link)

The public link above is the **hosted** app in `huggingface_space/`, deployed on **Streamlit
Community Cloud** (free). Instead of standing up a serving server in the cloud, it reads
recommendations **precomputed from the real models** into `huggingface_space/precomputed/`, so it
needs only `streamlit / pandas / numpy / pyarrow`.

To (re)deploy: push the repo, go to [share.streamlit.io](https://share.streamlit.io), pick the repo
and branch `main`, set **Main file path** to `huggingface_space/app.py`, and Deploy. To serve *live*
predictions instead, run TF Serving on a reachable host and point the local app's `RETRIEVAL_URL` /
`RANKER_URL` environment variables at it.
