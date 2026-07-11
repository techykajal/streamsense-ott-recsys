#!/usr/bin/env bash
# Package the PyTorch ranker into a .mar and serve with TorchServe (works on Colab).
set -e

mkdir -p model_store
torch-model-archiver \
  --model-name ranker \
  --version 1.0 \
  --serialized-file artifacts/ranker.pt \
  --handler serving/torchserve_handler.py \
  --extra-files "src/ranking_torch.py,artifacts/id_maps.json" \
  --export-path model_store -f

nohup torchserve --start --ncs \
  --model-store model_store \
  --models ranker=ranker.mar \
  --disable-token-auth > /tmp/torchserve.log 2>&1 &
sleep 15
echo "TorchServe up on :8080"

# sample request
curl -s -X POST http://localhost:8080/predictions/ranker \
  -H "Content-Type: application/json" \
  -d '{"user": 0, "movie": 12, "seg": 3}'
echo
