# StreamSense Explorer — hosted demo

Interactive two-stage OTT recommender demo (Streamlit Community Cloud).

Pick a user for live recommendations, or a movie for similar titles and likely viewers.
Two modes: **One Stage Ranker (TFX)** and **Two Stage (Retrieval: TF + Ranker: PyTorch)**.

The recommendations were **precomputed from the real trained models** (two-tower retrieval,
PyTorch ranker, TFX ranker) and are read from `precomputed/`. The hosted app therefore needs
only `streamlit / pandas / numpy / pyarrow` — no TensorFlow or PyTorch at runtime, so it starts
fast and stays within the free tier's memory. Content-similar titles are computed live from the
movie embeddings.

The full training pipeline (TFX), per-model evaluation, and the production serving runtimes
(TF Serving / TorchServe / NVIDIA Triton) live in the main repository.

## Deploy (Streamlit Community Cloud)
Main file path: `huggingface_space/app.py` — reads `data/processed/` and `metrics/` from the
repo root and `precomputed/` from this folder.
