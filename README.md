# StreamSense — OTT Content Personalization (2-Day MLOps POC)

A two-stage recommender (retrieval + ranking) for OTT content, built to demonstrate
**every framework in the target JD**: TFX, Kubeflow Pipelines SDK, TensorFlow Serving,
NVIDIA Triton, and TorchServe — plus TensorFlow, PyTorch, recommendation/ranking models,
user segmentation, content analysis (GenAI embeddings), and offline A/B evaluation.

Runs end-to-end on **Google Colab (free T4 GPU)**. Laptop/Docker notes included where Colab is limited.

---

## The idea in one line
> Given a user's viewing history, generate personalized OTT recommendations using a
> **two-tower retrieval** model to shortlist candidates and a **ranking** model to order them,
> then serve each stage on production-grade infra (TF Serving, TorchServe, Triton) orchestrated by a **TFX pipeline** compiled for **Kubeflow**.

## Why this maps to the JD
| JD requirement | Where it shows up here |
|---|---|
| Recommendation engines, ranking algorithms | Two-tower retrieval (TFRS) + PyTorch/TF ranker |
| User segmentation | K-Means on user embeddings → `segment` feature |
| Content analysis | Sentence-transformer / GenAI embeddings of titles+overviews |
| Data pipelines, feature engineering | TFX `Transform` + `features.py` |
| TFX, Kubeflow Pipelines SDK | `tfx_pipeline.py` run locally + compiled to KFP v2 IR |
| TensorFlow Serving, Triton, TorchServe | `serving/` — all three serve a model |
| Python, TensorFlow, PyTorch | retrieval in TF, ranker in PyTorch |
| A/B tests & offline experiments | `ab_eval.py` (recall@k, NDCG@k across 2 variants) |
| Generative AI in production | LLM embeddings as features + "explain this rec" layer |
| Real-time streaming (design) | streaming-feature design in the plan doc |

---

## Two-stage architecture
```
                 ┌─────────────────────── TFX PIPELINE (LocalDagRunner → compiled to KFP v2) ───────────────────────┐
                 │ ExampleGen → StatisticsGen → SchemaGen → ExampleValidator → Transform → Trainer → Evaluator → Pusher │
                 └────────────────────────────────────────────────────────────────────────────────────────────────────┘

  user_id ─┐
           ▼
   ┌──────────────┐   top-K candidates   ┌──────────────┐   ordered list   ┌─────────────┐
   │  RETRIEVAL   │ ───────────────────▶ │   RANKING    │ ───────────────▶ │  RESPONSE   │
   │ two-tower TF │                      │ PyTorch MLP  │                  │ + GenAI     │
   │ (TFRS)       │                      │ +embeddings  │                  │ explanation │
   └──────┬───────┘                      └──────┬───────┘                  └─────────────┘
          │ SavedModel                          │ TorchScript / ONNX
          ▼                                      ▼
   ┌──────────────┐                      ┌──────────────┬──────────────┐
   │ TF SERVING   │                      │  TORCHSERVE  │    TRITON    │
   │ REST/gRPC    │                      │   (.mar)     │ (ONNX/TRT)   │
   └──────────────┘                      └──────────────┴──────────────┘
```

---

## Repo layout
```
streamsense-ott-recsys/
├── requirements.txt
├── src/
│   ├── features.py            # data load, feature engineering, user segmentation, content embeddings
│   ├── two_tower.py           # TFRS retrieval model (query + candidate towers)
│   ├── tfx_pipeline.py        # TFX pipeline (local run) + KFP v2 compile
│   ├── ranking_torch.py       # PyTorch ranking model + training loop + TorchScript export
│   ├── export_onnx.py         # PyTorch → ONNX for Triton
│   └── ab_eval.py             # offline A/B eval: recall@k, NDCG@k
├── serving/
│   ├── tf_serving_run.sh      # install + launch TF Serving in Colab, sample request
│   ├── torchserve_handler.py  # custom TorchServe handler
│   └── torchserve_run.sh      # archive .mar + launch + sample request
└── triton/
    ├── run_triton.sh          # launch Triton (Docker on laptop / NGC binary on Colab) + client
    └── model_repository/
        └── ranker_onnx/
            ├── config.pbtxt   # Triton model config (dynamic batching)
            └── 1/             # put model.onnx here (produced by export_onnx.py)
```

---

## Day-by-day (each block ≈ 2–3 hrs)

### Day 1 — Retrieval + TFX + TF Serving
1. `features.py` — download MovieLens (`ml-latest-small` for speed, `ml-25m` if you have time), build user/item features, K-Means `segment`, sentence-transformer content embeddings.
2. `two_tower.py` — train TFRS retrieval model, check `recall@k`, export SavedModel.
3. `tfx_pipeline.py` — run the same training inside a **TFX** pipeline with `LocalDagRunner` (ExampleGen…Pusher).
4. `serving/tf_serving_run.sh` — launch **TensorFlow Serving**, hit the REST endpoint.

### Day 2 — Ranking + TorchServe + Triton + Kubeflow + A/B
5. `ranking_torch.py` — train **PyTorch** ranker, export TorchScript.
6. `serving/torchserve_*` — package `.mar`, launch **TorchServe**, query.
7. `export_onnx.py` + `triton/` — export ONNX, launch **Triton**, compare latency & dynamic batching.
8. `tfx_pipeline.py --compile-kfp` — compile the pipeline to **Kubeflow Pipelines v2** IR YAML.
9. `ab_eval.py` — offline A/B: retrieval-only vs retrieval+ranking → recall@k / NDCG@k table.

---

## Quick start (Colab)
```bash
!git clone <your-repo> && cd streamsense-ott-recsys
!pip install -r requirements.txt
!python src/features.py            # → data/ features
!python src/two_tower.py           # → artifacts/retrieval SavedModel
!python src/tfx_pipeline.py        # → TFX run (local)
!bash  serving/tf_serving_run.sh   # → TF Serving on :8501
!python src/ranking_torch.py       # → artifacts/ranker.pt (+ TorchScript)
!bash  serving/torchserve_run.sh   # → TorchServe on :8080
!python src/export_onnx.py         # → triton/.../1/model.onnx
!bash  triton/run_triton.sh        # → Triton on :8000
!python src/tfx_pipeline.py --compile-kfp   # → ott_pipeline.yaml (KFP v2)
!python src/ab_eval.py             # → metrics table
```

## Honest Colab caveats (say these in the interview, they show maturity)
- **TFX/TFRS pin TensorFlow versions.** Use a fresh Colab runtime and the pins in `requirements.txt`; restart runtime after install.
- **Triton is Docker-first.** On a laptop use the NGC Docker image (`run_triton.sh` default). On Colab (no Docker), run the `tritonserver` binary from the NGC tarball, or fall back to ONNX Runtime + the committed `config.pbtxt` to prove you understand the deployment (batching, instance groups, backends).
- **Kubeflow full cluster** isn't on Colab. We **author + compile** the pipeline to KFP v2 IR (real, reviewable artifact) and optionally run `kfp local`. Note in interview: "same DAG runs on Vertex AI Pipelines / GKE by swapping the runner."
