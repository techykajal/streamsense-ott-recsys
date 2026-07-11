"""
torchserve_handler.py — custom TorchServe handler for the PyTorch ranker.

Input JSON:  {"user": 0, "movie": 12, "seg": 3}  (or a list of such objects for batching)
Output JSON: [{"score": 0.83}, ...]
"""
import json
import torch
from ts.torch_handler.base_handler import BaseHandler
from ranking_torch import Ranker


class RankerHandler(BaseHandler):
    def initialize(self, ctx):
        props = ctx.system_properties
        model_dir = props.get("model_dir")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(f"{model_dir}/ranker.pt", map_location=self.device)
        self.model = Ranker(ck["n_user"], ck["n_movie"], ck["n_seg"], ck["dim"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.initialized = True

    def preprocess(self, data):
        body = data[0].get("body") or data[0].get("data")
        if isinstance(body, (bytes, bytearray)):
            body = json.loads(body)
        rows = body if isinstance(body, list) else [body]
        t = lambda k: torch.tensor([int(r[k]) for r in rows], device=self.device)
        return t("user"), t("movie"), t("seg")

    def inference(self, x):
        with torch.no_grad():
            return self.model(*x)

    def postprocess(self, out):
        return [[{"score": float(s)} for s in out.cpu().tolist()]]
