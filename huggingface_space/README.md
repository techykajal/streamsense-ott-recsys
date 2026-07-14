---
title: StreamSense Explorer
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# StreamSense Explorer

Interactive demo of a two-stage OTT recommender (retrieval + ranking). Pick a user to get live
recommendations, or a movie to see similar titles and likely viewers. Two modes:
**One Stage Ranker (TFX)** and **Two Stage (Retrieval: TF + Ranker: PyTorch)**.

Models are loaded in-process (TensorFlow + PyTorch). Full project, TFX training pipeline, evaluation,
and the production serving runtimes (TF Serving / TorchServe / Triton) live in the main repo.
