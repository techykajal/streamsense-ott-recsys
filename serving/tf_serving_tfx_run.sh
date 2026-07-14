#!/usr/bin/env bash
# TensorFlow Serving for the *TFX-trained ranker* SavedModel (artifacts/ranking_tf).
# This is a SEPARATE serving process from the two-tower retrieval one (tf_serving_run.sh);
# run both if you want retrieval + ranking served side by side.
#
# The TFX Pusher already writes a versioned directory (artifacts/ranking_tf/<version>/),
# which is exactly the layout TF Serving expects, so we mount it directly.
#
# The model's serving_default signature takes a serialized tf.Example under the input
# "examples" and returns "prediction" (P(engagement)). Build the request with
# serving/make_ranker_request.py (needs TensorFlow).
set -e

MODEL_SRC="${MODEL_SRC:-$(pwd)/artifacts/ranking_tf}"     # contains <version>/saved_model.pb
PORT="${PORT:-8502}"                                      # 8502 so it can coexist with retrieval on 8501

PLATFORM=""
if [ "$(uname -m)" = "arm64" ]; then PLATFORM="--platform linux/amd64"; fi

docker rm -f tfserving-ranker >/dev/null 2>&1 || true

echo "Starting TensorFlow Serving for the TFX ranker on port ${PORT} ..."
docker run -d --name tfserving-ranker $PLATFORM \
  -p ${PORT}:8501 \
  -v "${MODEL_SRC}:/models/ranker" \
  -e MODEL_NAME=ranker \
  tensorflow/serving

for i in $(seq 1 30); do
  if curl -s http://localhost:${PORT}/v1/models/ranker >/dev/null 2>&1; then break; fi
  sleep 2
done
echo "TFX ranker up on http://localhost:${PORT}/v1/models/ranker"
echo

# Sample request (user_id=1, movie_id=1, segment=0). Requires TensorFlow to serialize tf.Example.
if python -c "import tensorflow" 2>/dev/null; then
  REQ=$(python serving/make_ranker_request.py 1 1 0)
  echo "Sample prediction:"
  curl -s -X POST http://localhost:${PORT}/v1/models/ranker:predict -d "$REQ" | head -c 400
  echo
else
  echo "(install tensorflow to auto-build a sample tf.Example request; see serving/make_ranker_request.py)"
fi
echo
echo "Stop it later with:  docker rm -f tfserving-ranker"
