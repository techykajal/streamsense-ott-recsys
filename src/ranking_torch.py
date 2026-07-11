"""
ranking_torch.py — PyTorch ranking model (the one TorchServe + Triton will serve).

A compact DLRM-style ranker: user/movie/segment embeddings -> MLP -> P(engagement).
Exports:
  * artifacts/ranker.pt          (state_dict + id maps, for TorchServe handler)
  * artifacts/ranker_scripted.pt (TorchScript, optional)
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ART = "artifacts"
DIM = 32


class Ranker(nn.Module):
    def __init__(self, n_user, n_movie, n_seg, dim=DIM):
        super().__init__()
        self.u = nn.Embedding(n_user, dim)
        self.m = nn.Embedding(n_movie, dim)
        self.s = nn.Embedding(n_seg, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, user, movie, seg):
        x = torch.cat([self.u(user), self.m(movie), self.s(seg)], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


def load():
    df = pd.read_parquet("data/processed/interactions.parquet")
    uid = {v: i for i, v in enumerate(df.user_id.unique())}
    mid = {v: i for i, v in enumerate(df.movie_id.unique())}
    X = np.stack([
        df.user_id.map(uid).values,
        df.movie_id.map(mid).values,
        df.segment.values,
    ], axis=1).astype("int64")
    y = df.label.values.astype("float32")
    return X, y, uid, mid, int(df.segment.max()) + 1


def main():
    X, y, uid, mid, n_seg = load()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=4096, shuffle=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ranker(len(uid), len(mid), n_seg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = nn.BCELoss()

    model.train()
    for epoch in range(3):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            p = model(xb[:, 0], xb[:, 1], xb[:, 2])
            loss = lossf(p, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(yb)
        print(f"epoch {epoch}  loss={tot/len(ds):.4f}")

    os.makedirs(ART, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "n_user": len(uid), "n_movie": len(mid), "n_seg": n_seg,
                "dim": DIM}, f"{ART}/ranker.pt")
    json.dump({"uid": uid, "mid": mid}, open(f"{ART}/id_maps.json", "w"))

    # TorchScript for a portable, Python-free artifact
    model.eval().cpu()
    example = (torch.zeros(1, dtype=torch.long),) * 3
    torch.jit.save(torch.jit.trace(model, example), f"{ART}/ranker_scripted.pt")
    print(f"Saved ranker → {ART}/ranker.pt (+ TorchScript, id_maps.json)")


if __name__ == "__main__":
    main()
