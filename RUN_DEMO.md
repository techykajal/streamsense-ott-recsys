# Running the live demo — serving + interactive dashboard

This brings up the **real serving path** and the **StreamSense Explorer** dashboard, which issues
live requests to it. The dashboard never loads a model in-process. It offers two selectable
strategies so you can demonstrate the scale trade-off:

```
 TESTING  (small catalogue — one ranker is enough)
   user ─────────────────────────────▶ TFX ranker (TF Serving :8502) ─▶ scores ─▶ top-K

 PROD  (millions-of-items pattern — two stages)
   user ─▶ Retrieval (TF Serving :8501) ─▶ top-K candidates
                                              └▶ PyTorch ranker (TorchServe :8080) ─▶ scores ─▶ top-K
```

| Endpoint | Model | Used by | Script |
|---|---|---|---|
| `:8501` | two-tower retrieval | Prod | `serving/tf_serving_run.sh` |
| `:8502` | TFX/Keras ranker | Testing | `serving/tf_serving_tfx_run.sh` |
| `:8080` | PyTorch ranker | Prod | `serving/torchserve_run.sh` |

## Prerequisites

1. Build the data and models once:
   ```bash
   python src/features.py        # data/processed/*.parquet + interactions.tfrecord
   python src/two_tower.py        # retrieval SavedModel  -> artifacts/retrieval
   python src/ranking_torch.py    # PyTorch ranker        -> artifacts/ranker.pt
   # TFX ranker: run notebooks/StreamSense_Colab_TFX.ipynb -> artifacts/ranking_tf/<version>/
   ```
2. Docker running (for TF Serving), or use the Colab flow below.

## Step 1 — Start the endpoints you need

For **Testing** mode you only need the TFX ranker. For **Prod** mode you need retrieval +
TorchServe. Start all three to switch freely in the dashboard:

```bash
bash serving/tf_serving_tfx_run.sh    # TFX ranker      -> :8502   (Testing)
bash serving/tf_serving_run.sh        # two-tower       -> :8501   (Prod)
bash serving/torchserve_run.sh        # PyTorch ranker  -> :8080   (Prod)
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
# after cloning the repo, building data/models, and starting both TF Serving containers:
!pip install -q streamlit
!npm install -g localtunnel
# launch streamlit in the background, then open the tunnel
import subprocess, time
subprocess.Popen(["streamlit","run","app/streamsense_explorer.py","--server.port","8500"])
time.sleep(6)
!npx localtunnel --port 8500       # prints a public https URL for the dashboard
```

> For a permanent free host, deploy the dashboard on **Hugging Face Spaces** or **Streamlit
> Community Cloud**. In that setup, run TF Serving on a reachable host (e.g., Google Cloud Run's free
> tier) and point `RETRIEVAL_URL` / `RANKER_URL` (environment variables) at it.
