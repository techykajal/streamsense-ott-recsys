# Running StreamSense in Docker (the fix for Apple Silicon)

TFX won't `pip install` on an Apple Silicon Mac (no arm64 build of `ml-metadata`, and TFX vs
TensorFlow-Recommenders demand conflicting TensorFlow packages on macOS). So we run the code
inside a **Linux container**, where everything installs cleanly. You already have Docker Desktop.

You can stop fighting the conda/venv install for the TensorFlow side — the container replaces it.

---

## One-time: build the image
From the project root (where the `Dockerfile` is), with Docker Desktop running:
```bash
docker build -t streamsense .
```
First build takes ~10–15 min (it downloads TensorFlow, TFX, PyTorch). It's cached after that.
> On Apple Silicon you may see a "platform linux/amd64 does not match host" note — that's expected;
> it's the emulation we want. Let it finish.

---

## Run the training / pipeline scripts inside the container
Open an interactive shell in the container, with your project folder mounted so files you edit in
VSCode are visible inside, and outputs land back on your Mac:
```bash
docker run -it --rm -v "$PWD":/app streamsense bash
```
Your prompt changes to something like `root@abc123:/app#`. Now run the pipeline (Phases 1 & 4):
```bash
python src/features.py         # downloads MovieLens, builds features  → data/
python src/two_tower.py        # retrieval model                        → artifacts/retrieval/
python src/tfx_pipeline.py     # the TFX pipeline (this is what failed before — works here)
python src/ranking_torch.py    # PyTorch ranker                         → artifacts/ranker.pt
python src/export_onnx.py      # ONNX export for Triton
python src/tfx_pipeline.py --compile-kfp   # Kubeflow v2 spec → ott_pipeline.yaml
python src/ab_eval.py          # Recall@K / NDCG@K table
exit                           # leave the container (files remain on your Mac)
```
Because of the `-v "$PWD":/app` mount, everything written to `data/`, `artifacts/`, etc. appears in
your project folder on the Mac.

---

## Serving stays on the host
The serving scripts already launch their own containers, so run these from your **normal Mac
Terminal** (not inside the streamsense container), after the artifacts exist:
```bash
bash serving/tf_serving_run.sh     # TF Serving  (Docker)  :8501
bash serving/torchserve_run.sh     # TorchServe            :8080   (needs torch on host — see note)
bash triton/run_triton.sh          # Triton      (Docker)  :8000   (ONNX fallback if it won't start)
```
> TorchServe note: to run `torchserve_run.sh` on the host you'd need torch installed on the Mac.
> Easiest is to also run TorchServe from inside the container instead:
> `docker run -it --rm -v "$PWD":/app -p 8080:8080 streamsense bash` then `bash serving/torchserve_run.sh`.

---

## Smoother option: VSCode Dev Containers
If you'd like VSCode to run *inside* the container automatically (integrated terminal, IntelliSense,
no manual `docker run`):
1. Install the **Dev Containers** extension in VSCode.
2. `Cmd+Shift+P` → "Dev Containers: Reopen in Container" → it uses this `Dockerfile`.
Then every terminal and run is already in the Linux environment.

---

## Don't want Docker for the TF parts at all?
Alternative: run `features.py`, `two_tower.py`, `tfx_pipeline.py` in **Google Colab** (Linux, free),
and keep the PyTorch parts native on your Mac. The container path above is cleaner for a single
reproducible setup, but Colab is a valid fallback if the build is too heavy.

---

## Commit these to GitHub
```bash
git add Dockerfile .dockerignore DOCKER_SETUP.md
git commit -m "Add Docker setup for reproducible Linux environment"
git push
```
