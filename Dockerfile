FROM tensorflow/tfx:1.15.1

WORKDIR /app

# The TFX image pins its deps via a hash-locked PIP_CONSTRAINT file. One of those
# hashes is stale vs PyPI, which blocks extra installs — so unset it for our layer.
ENV PIP_CONSTRAINT=""

# Only the extras NOT already in the TFX image: the PyTorch ranker, its serving tools,
# the retrieval library, and the content-embedding model.
RUN pip install --timeout 180 --retries 6 \
      tensorflow-recommenders==0.7.3 \
      torch==2.3.1 \
      torchserve==0.11.1 \
      torch-model-archiver==0.11.1 \
      onnx==1.16.1 \
      onnxruntime==1.18.1 \
      sentence-transformers==2.7.0

COPY . .

CMD ["bash"]