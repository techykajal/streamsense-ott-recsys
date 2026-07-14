#!/usr/bin/env bash
# One-command local demo (macOS / Linux, no Docker). Loads the committed models natively and
# serves them, then launches the dashboard pointed at the local API.
set -e
cd "$(dirname "$0")/.."          # repo root

python serving/local_api.py &     # native serving on :8000
API_PID=$!
echo "local serving starting (pid $API_PID) ..."
# wait for the API to answer
for i in $(seq 1 40); do
  curl -s http://localhost:8000/ping >/dev/null 2>&1 && break
  sleep 1
done
echo "serving up on http://localhost:8000"

export RETRIEVAL_URL=http://localhost:8000/v1/models/retrieval:predict
export RANKER_URL=http://localhost:8000/v1/models/ranker:predict
export TORCH_URL=http://localhost:8000/predictions/ranker

streamlit run app/streamsense_explorer.py     # opens http://localhost:8501

kill $API_PID 2>/dev/null || true
