#!/usr/bin/env bash
# TensorFlow Serving for the retrieval SavedModel — via Docker (macOS / Linux).
# Requires Docker Desktop running. On Apple Silicon we force linux/amd64 (emulated).
set -e

MODEL_SRC="${MODEL_SRC:-$(pwd)/artifacts/retrieval}"
SERVE_DIR="$(pwd)/artifacts/tfserving/retrieval/1"   # TF Serving needs <name>/<version>/

# Stage a versioned copy of the SavedModel.
mkdir -p "$SERVE_DIR"
cp -r "$MODEL_SRC"/* "$SERVE_DIR/"

# Pick platform flag automatically (Apple Silicon => amd64 emulation).
PLATFORM=""
if [ "$(uname -m)" = "arm64" ]; then PLATFORM="--platform linux/amd64"; fi

# Remove any previous container with the same name.
docker rm -f tfserving-retrieval >/dev/null 2>&1 || true

echo "Starting TensorFlow Serving (first pull can take a few minutes)..."
docker run -d --name tfserving-retrieval $PLATFORM \
  -p 8501:8501 \
  -v "$(pwd)/artifacts/tfserving/retrieval:/models/retrieval" \
  -e MODEL_NAME=retrieval \
  tensorflow/serving

# Wait for the REST API to come up.
for i in $(seq 1 30); do
  if curl -s http://localhost:8501/v1/models/retrieval >/dev/null 2>&1; then break; fi
  sleep 2
done
echo "TF Serving up on http://localhost:8501"

# Sample request: top-K movies for user_id=1, segment=0.
curl -s -X POST http://localhost:8501/v1/models/retrieval:predict \
  -d '{"inputs": {"user_id": ["1"], "segment": [0]}}' | head -c 800
echo
echo
echo "Stop it later with:  docker rm -f tfserving-retrieval"
