#!/usr/bin/env bash
# Launch NVIDIA Triton serving the ONNX ranker, then query it.
#
# LAPTOP (recommended, has Docker):
#   docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
#     -v "$(pwd)/triton/model_repository:/models" \
#     nvcr.io/nvidia/tritonserver:24.05-py3 \
#     tritonserver --model-repository=/models
#
# COLAB (no Docker): download the tritonserver binary tarball from NGC and run it,
#   or if that is too heavy, skip the server and validate with onnxruntime
#   (export_onnx.py already does a parity check) — the config.pbtxt still proves
#   you can configure Triton (batching, instance groups, backend).
set -e
MODE="${1:-docker}"

if [ "$MODE" = "docker" ]; then
  # Apple Silicon: Triton images are amd64 → emulate. May be slow; use ONNX fallback if it fails.
  PLATFORM=""
  if [ "$(uname -m)" = "arm64" ]; then PLATFORM="--platform linux/amd64"; fi
  docker rm -f triton-ranker >/dev/null 2>&1 || true
  docker run --rm -d --name triton-ranker $PLATFORM \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    -v "$(pwd)/triton/model_repository:/models" \
    nvcr.io/nvidia/tritonserver:24.05-py3 \
    tritonserver --model-repository=/models
  echo "Waiting for Triton to load..."; sleep 15
fi

# --- client query (Python) ---
python - <<'PY'
import numpy as np, tritonclient.http as http
c = http.InferenceServerClient("localhost:8000")
def col(name, val):
    t = http.InferInput(name, [1, 1], "INT64")
    t.set_data_from_numpy(np.array([[val]], dtype=np.int64))
    return t
ins = [col("user", 0), col("movie", 12), col("seg", 3)]
out = http.InferRequestedOutput("score")
r = c.infer("ranker_onnx", ins, outputs=[out])
print("Triton score:", r.as_numpy("score").ravel())
PY
