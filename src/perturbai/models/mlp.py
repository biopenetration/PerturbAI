"""PyTorch MLP: perturbation features -> delta expression. Early stopping on held-out PERTURBATIONS."""
from __future__ import annotations

import numpy as np

from perturbai.models.base import PerturbModel


class MLPModel(PerturbModel):
    name = "mlp"

    def __init__(self, hidden=(512, 512), dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=300,
                 batch_size=64, val_frac=0.15, patience=25, seed=0, device=None, verbose=False):
        self.__dict__.update(dict(hidden=tuple(hidden), dropout=dropout, lr=lr, weight_decay=weight_decay,
                                  epochs=epochs, batch_size=batch_size, val_frac=val_frac,
                                  patience=patience, seed=seed, device=device, verbose=verbose))
        self.history = []

    def _build(self, d_in, d_out):
        import torch.nn as nn
        layers, d = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    def fit(self, X, Y):
        import copy
        import torch
        torch.manual_seed(self.seed)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = dev
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-6
        self.y_scale = float(Y.std() + 1e-8)
        Xs = ((X - self.x_mu) / self.x_sd).astype(np.float32)
        Ys = (Y / self.y_scale).astype(np.float32)
        n = len(Xs)
        perm = np.random.default_rng(self.seed).permutation(n)
        nv = int(round(self.val_frac * n)) if n >= 20 else 0
        va, tr = perm[:nv], perm[nv:]
        Xt, Yt = torch.tensor(Xs[tr], device=dev), torch.tensor(Ys[tr], device=dev)
        Xv = torch.tensor(Xs[va], device=dev) if nv else None
        Yv = torch.tensor(Ys[va], device=dev) if nv else None
        self.net = self._build(Xs.shape[1], Ys.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for ep in range(self.epochs):
            self.net.train()
            p = torch.randperm(len(Xt), device=dev)
            tot = 0.0
            for i in range(0, len(Xt), self.batch_size):
                b = p[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b]), Yt[b])
                loss.backward(); opt.step()
                tot += loss.item() * len(b)
            tr_loss = tot / len(Xt)
            if nv:
                self.net.eval()
                with torch.no_grad():
                    v = loss_fn(self.net(Xv), Yv).item()
            else:
                v = tr_loss
            self.history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": v})
            if v < best - 1e-6:
                best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and ep % 20 == 0:
                print(f"epoch {ep} train {tr_loss:.4f} val {v:.4f}")
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        Xs = torch.tensor(((X - self.x_mu) / self.x_sd).astype(np.float32), device=self.dev)
        with torch.no_grad():
            return self.net(Xs).cpu().numpy() * self.y_scale

    def save(self, path):
        import torch
        torch.save({"state": self.net.state_dict(), "x_mu": self.x_mu, "x_sd": self.x_sd,
                    "y_scale": self.y_scale, "cfg": {"hidden": self.hidden, "dropout": self.dropout},
                    "d_in": len(self.x_mu), "d_out": self.net[-1].out_features}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        import torch
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(hidden=ck["cfg"]["hidden"], dropout=ck["cfg"]["dropout"], device=device)
        m.dev = device; m.x_mu, m.x_sd, m.y_scale = ck["x_mu"], ck["x_sd"], ck["y_scale"]
        m.net = m._build(ck["d_in"], ck["d_out"]).to(device); m.net.load_state_dict(ck["state"])
        return m
