"""
export_onnx.py — PyTorch ranker -> ONNX for NVIDIA Triton.

Produces triton/model_repository/ranker_onnx/1/model.onnx with dynamic batch axis.
Verifies parity with onnxruntime before handing off to Triton.
"""
import os
import numpy as np
import torch
from ranking_torch import Ranker

CKPT = "artifacts/ranker.pt"
OUT = "triton/model_repository/ranker_onnx/1/model.onnx"


def main():
    ck = torch.load(CKPT, map_location="cpu")
    model = Ranker(ck["n_user"], ck["n_movie"], ck["n_seg"], ck["dim"])
    model.load_state_dict(ck["state_dict"]); model.eval()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dummy = (torch.zeros(1, dtype=torch.long),) * 3
    torch.onnx.export(
        model, dummy, OUT,
        input_names=["user", "movie", "seg"], output_names=["score"],
        dynamic_axes={"user": {0: "B"}, "movie": {0: "B"},
                      "seg": {0: "B"}, "score": {0: "B"}},
        opset_version=17)
    print(f"Exported ONNX → {OUT}")

    # parity check
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    feed = {"user": np.array([0], np.int64),
            "movie": np.array([0], np.int64),
            "seg": np.array([0], np.int64)}
    onnx_out = sess.run(None, feed)[0]
    torch_out = model(torch.zeros(1, dtype=torch.long),
                      torch.zeros(1, dtype=torch.long),
                      torch.zeros(1, dtype=torch.long)).detach().numpy()
    print(f"parity |Δ| = {np.abs(onnx_out.ravel() - torch_out.ravel()).max():.2e}")


if __name__ == "__main__":
    main()
