# StreamSense — Build It From Scratch on macOS (absolute-beginner guide)

Follow this top to bottom. Every command is meant to be pasted into the macOS **Terminal**
(open it with `Cmd+Space` → type "Terminal" → Enter). Lines starting with `#` are comments —
you don't type those.

**Folder name (local AND GitHub):** `streamsense-ott-recsys`

---

## 0. The honest reality on a Mac (read once, then proceed)

You picked "full local with Docker" — good. Two truths so nothing surprises you:

1. **Python must be 3.10 or 3.11.** TFX 1.15 does **not** work on Python 3.12+. We install 3.10.
2. **TF Serving and Triton run inside Docker, not as Mac apps.** If your Mac is **Apple Silicon**
   (M1/M2/M3/M4 — check with `uname -m`, `arm64` = Apple Silicon), those Docker images are
   built for Intel (amd64) and run under emulation. TF Serving works fine that way.
   **Triton is heavy and may be slow/flaky under emulation** — if it won't start, use the
   built-in ONNX-Runtime fallback (Step 6, Phase 3) or run Triton in Colab. Everything else
   runs natively and fast. The committed `triton/config.pbtxt` still proves the skill either way.

We build in **4 phases** so you get working wins early:
`Phase 1` retrieval + TFX + TF Serving → `Phase 2` ranking + TorchServe →
`Phase 3` Triton → `Phase 4` Kubeflow compile + A/B eval.

---

## 1. Install prerequisites (do this before running ANY script)

### 1a. Homebrew (the macOS package manager)
```bash
# If you don't already have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# After it finishes, follow the "Next steps" it prints to add brew to your PATH, then:
brew --version
```

### 1b. Git
```bash
brew install git
git --version
```

### 1c. Python 3.10 via pyenv (keeps it separate from macOS system Python)
```bash
brew install pyenv
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
pyenv install 3.10.14
```

### 1d. Docker Desktop (for TF Serving + Triton)
Download and install from https://www.docker.com/products/docker-desktop/ , open the app once,
and leave it running. Then verify in Terminal:
```bash
docker --version
docker run --rm hello-world      # should print "Hello from Docker!"
```

### 1e. VSCode + Python extension
Install VSCode from https://code.visualstudio.com/ , open it, go to Extensions (the squares icon),
install **Python** (by Microsoft). That's all you need for now.

---

## 2. Create the project folder
```bash
cd ~/Documents            # or wherever you keep projects
mkdir streamsense-ott-recsys
cd streamsense-ott-recsys
pyenv local 3.10.14       # pins THIS folder to Python 3.10
python --version          # must say Python 3.10.14
```

---

## 3. Create and activate the virtual environment

A virtual environment ("venv") is a private, isolated copy of Python + libraries for this
project only, so it never clashes with anything else on your Mac.

```bash
python -m venv .venv          # creates a hidden .venv folder
source .venv/bin/activate     # ACTIVATE it — your prompt now shows (.venv)
python -m pip install --upgrade pip
```

- You know it worked when your Terminal prompt starts with `(.venv)`.
- **Every new Terminal session**, re-run `source .venv/bin/activate` before working.
- To leave it later: type `deactivate`.

---

## 4. Put the files in place (the directory tree)

Unzip the provided **`streamsense-ott-recsys.zip`** into this folder (or copy the files from the
download). You should end up with exactly this structure:

```
streamsense-ott-recsys/
├── requirements.txt          # 1st file that matters — the library list
├── .gitignore                # tells git what NOT to upload
├── README.md                 # architecture + quick reference
├── SETUP_GUIDE.md            # this guide
├── src/
│   ├── features.py           # Step 6 Phase 1 — data + features + segments + content embeddings
│   ├── two_tower.py          # Phase 1 — retrieval model (TensorFlow Recommenders)
│   ├── tfx_pipeline.py       # Phase 1/4 — TFX pipeline + Kubeflow compile
│   ├── trainer_module.py     # used by tfx_pipeline.py (don't run directly)
│   ├── ranking_torch.py      # Phase 2 — PyTorch ranking model
│   ├── export_onnx.py        # Phase 3 — PyTorch → ONNX for Triton
│   └── ab_eval.py            # Phase 4 — offline A/B: Recall@K, NDCG@K
├── serving/
│   ├── tf_serving_run.sh     # Phase 1 — launch TF Serving (Docker)
│   ├── torchserve_handler.py # Phase 2 — TorchServe request handler
│   └── torchserve_run.sh     # Phase 2 — package .mar + launch TorchServe
└── triton/
    ├── run_triton.sh         # Phase 3 — launch Triton (Docker) + query
    └── model_repository/
        └── ranker_onnx/
            ├── config.pbtxt  # Triton model config (dynamic batching)
            └── 1/            # export_onnx.py drops model.onnx here
```

> **What order do files "come in"?** Conceptually: `requirements.txt` first (so you can install),
> then you *run* the `src/` scripts in Phase order (features → two_tower → tfx_pipeline →
> ranking_torch → export_onnx → ab_eval), calling the `serving/` and `triton/` scripts as each
> model becomes available. You don't have to create them one-by-one — the zip has them all;
> the phases below tell you which to run when.

---

## 5. Install the libraries (one command)

With the venv **active** (prompt shows `(.venv)`):
```bash
pip install -r requirements.txt
```
This takes a few minutes. Then sanity-check the big three imports:
```bash
python -c "import tensorflow as tf, tfx, torch; print('TF', tf.__version__, '| torch', torch.__version__)"
```
If that prints versions with no red errors, you're ready. (See Troubleshooting if not.)

---

## 6. Build & run — phase by phase

> Keep the venv active and stay in the project root folder for every command.
> `chmod +x serving/*.sh triton/*.sh` once so the shell scripts are runnable.

### Phase 1 — Retrieval + TFX + TensorFlow Serving
```bash
python src/features.py        # downloads MovieLens, builds features → data/processed/
python src/two_tower.py       # trains retrieval model → artifacts/retrieval/ (a SavedModel)
python src/tfx_pipeline.py    # runs the TFX pipeline locally → pushes ranking model
bash  serving/tf_serving_run.sh   # starts TF Serving in Docker on port 8501 + sends a test query
```
Expected: `features.py` prints counts of interactions/users/movies; `two_tower.py` prints falling
loss + a recall metric; TF Serving prints a JSON list of recommended movie ids.
> The `.sh` script uses Docker: `docker run ... tensorflow/serving`. On Apple Silicon add
> `--platform linux/amd64` (already handled in the script) — first run pulls the image, be patient.

### Phase 2 — Ranking + TorchServe
```bash
python src/ranking_torch.py   # trains PyTorch ranker → artifacts/ranker.pt (+ TorchScript)
bash  serving/torchserve_run.sh   # packages ranker.mar, starts TorchServe on :8080 + test query
```
Expected: falling BCE loss, then a JSON `{"score": 0.xx}` from the TorchServe endpoint.
TorchServe runs **natively** (no Docker) — this phase is the smoothest.

### Phase 3 — Triton (the Apple-Silicon-sensitive one)
```bash
python src/export_onnx.py     # PyTorch → triton/model_repository/ranker_onnx/1/model.onnx (prints parity ~1e-6)
bash  triton/run_triton.sh    # starts Triton in Docker on :8000 + Python client query
```
If Triton **won't start** on Apple Silicon (emulation can choke), don't fight it:
```bash
# Fallback that proves the same ONNX model serves correctly, no Docker:
python -c "import onnxruntime as ort, numpy as np; s=ort.InferenceSession('triton/model_repository/ranker_onnx/1/model.onnx'); \
print(s.run(None,{'user':np.array([0]),'movie':np.array([12]),'seg':np.array([3])}))"
```
Keep `config.pbtxt` in the repo — in interviews you explain the dynamic-batching config; running
Triton itself is best shown on Colab or a Linux/GPU box.

### Phase 4 — Kubeflow compile + offline A/B
```bash
python src/tfx_pipeline.py --compile-kfp   # → ott_pipeline.yaml (Kubeflow Pipelines v2 spec)
python src/ab_eval.py                       # prints Recall@10 / NDCG@10 for two variants + lift %
```
`ott_pipeline.yaml` is a real, reviewable artifact — it's what would run on Vertex AI / GKE.

---

## 7. Create the GitHub repo and push (step by step)

### 7a. Make the empty repo on GitHub (browser)
1. Go to https://github.com/new
2. **Repository name:** `streamsense-ott-recsys`
3. Description: `Two-stage OTT recommender (retrieval + ranking) with TFX, Kubeflow, TF Serving, Triton, TorchServe`
4. Choose **Public** (so interviewers can see it).
5. **Do NOT** tick "Add a README", ".gitignore", or "license" — you already have them locally.
6. Click **Create repository**. Leave that page open; you'll copy the URL it shows.

### 7b. Turn your local folder into a git repo and push
Run these in the project root, venv can stay active:
```bash
git init
git add .
git commit -m "Initial commit: StreamSense OTT recommender scaffold"
git branch -M main
# Use the URL from step 7a. HTTPS example:
git remote add origin https://github.com/<your-username>/streamsense-ott-recsys.git
git push -u origin main
```
If GitHub asks for a password, it actually wants a **Personal Access Token**, not your login
password: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained token →
give it `repo` access → copy it → paste it as the password. (Or install `brew install gh` and run
`gh auth login` once to skip this.)

### 7c. Your ongoing workflow (edit → save → push)
Every time you finish a piece:
```bash
git add src/two_tower.py           # or `git add .` for everything changed
git commit -m "Update two-tower retrieval model"
git push
```
Because of `.gitignore`, the big `data/`, `artifacts/`, `.venv/` folders are **not** uploaded —
only your code. That's intentional and correct.

---

## 8. Troubleshooting (common macOS issues)

| Symptom | Fix |
|---|---|
| `python --version` isn't 3.10 | Re-run `pyenv local 3.10.14` in the project folder; reopen Terminal. |
| `pip install` fails building `tfx`/`apache-beam` | Confirm Python is 3.10 (not 3.12). `pip install --upgrade pip setuptools wheel` then retry. |
| `command not found: pyenv` | Re-run the `echo 'eval "$(pyenv init -)"' >> ~/.zshrc` line, then `source ~/.zshrc`. |
| Docker command hangs / "cannot connect" | Open the Docker Desktop app and wait until its whale icon is steady, then retry. |
| TF Serving image won't run on M-series | The script passes `--platform linux/amd64`; first pull is slow, let it finish. |
| Triton container exits immediately | Use the ONNX-Runtime fallback in Phase 3; run real Triton on Colab/Linux. |
| `(.venv)` not in prompt | You forgot `source .venv/bin/activate` in this Terminal. |
| Push rejected / auth failed | Use a Personal Access Token as the password, or `gh auth login`. |

---

### One-glance daily routine
```bash
cd ~/Documents/streamsense-ott-recsys
source .venv/bin/activate
# ...work on a file, run it...
git add . && git commit -m "message" && git push
```
That's the whole loop. Build one phase, get it green, commit, push. Repeat.
