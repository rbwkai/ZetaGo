from __future__ import annotations

"""Convolutional autoencoder for Track B (EXECUTION_Phase3.md task 3.1).

Encoder is the champion CNN's body stack verbatim (F3): three same-padded 3x3
convs at `hidden` channels, so a trained encoder's conv weights can be copied
straight into a fresh `CNNModel` body for task 3.6. The decoder mirrors it
structurally; since every layer is stride-1/same-padded (no pooling), there is
no spatial shape to undo, so "mirrors" means the same conv-ReLU stack run in
reverse from `hidden` channels back to `in_channels`, not literal transposed
convolutions.

Loss is BCE-with-logits on the stone-occupancy planes (task 3.1: "pick one,
state it, don't report both") -- the reconstruction target is strictly {0,1}
per cell, which is what BCE assumes and MSE does not.

Reconstruction loss only: `fit` never receives anything but `states` arrays
(SS3.B); model selection uses the holdout reconstruction loss computed here,
nothing else.
"""

import os
from typing import List, Tuple

import numpy as np


class ConvAutoencoder:
    name = "conv_ae"

    def __init__(
        self,
        in_channels: int = 2,
        hidden: int = 64,
        latent: int = 64,
        board_size: int = 7,
        epochs: int = 150,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("PyTorch is required for ConvAutoencoder") from exc

        self.torch = torch
        self.nn = nn
        self.in_channels = in_channels
        self.hidden = hidden
        self.latent = latent
        self.board_size = board_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device

        # Seed before construction: weight initialisation happens inside
        # _build, so seeding only in fit() would leave init unseeded (same
        # fix as CNNModel -- see cnn_model.py).
        torch.manual_seed(seed)
        self.encoder, self.decoder = self._build(in_channels, hidden, latent, board_size)
        self.encoder.to(device)
        self.decoder.to(device)
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _build(self, in_channels: int, hidden: int, latent: int, board_size: int):
        nn = self.nn
        flat = hidden * board_size * board_size

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.to_latent = nn.Linear(flat, latent)

            def forward(self, x):
                z = self.body(x).flatten(1)
                return self.to_latent(z)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.from_latent = nn.Linear(latent, flat)
                self.body = nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1),
                    # raw logits -- BCEWithLogitsLoss applied outside
                )

            def forward(self, z):
                x = self.from_latent(z).view(-1, hidden, board_size, board_size)
                return self.body(x)

        return Encoder(), Decoder()

    def _loader(self, states: np.ndarray, shuffle: bool):
        torch = self.torch
        x = torch.from_numpy(np.asarray(states, dtype=np.float32))
        ds = torch.utils.data.TensorDataset(x)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def reconstruction_loss(self, states: np.ndarray) -> float:
        """Mean BCE-with-logits reconstruction loss over `states`, eval mode, no grad."""
        torch = self.torch
        self.encoder.eval()
        self.decoder.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for (xb,) in self._loader(states, shuffle=False):
                xb = xb.to(self.device)
                logits = self.decoder(self.encoder(xb))
                loss = self.loss_fn(logits, xb)
                total += float(loss.item()) * len(xb)
                n += len(xb)
        return total / max(1, n)

    def fit(self, fit_states: np.ndarray, holdout_states: np.ndarray) -> List[dict]:
        """Train on `fit_states`; `holdout_states` is scored every epoch for
        model selection (SS3.B) but never trained on. Returns the per-epoch
        (train_loss, holdout_loss) history."""
        torch = self.torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        dl = self._loader(fit_states, shuffle=True)
        history = []
        for ep in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            loss_sum, n_batches = 0.0, 0
            for (xb,) in dl:
                xb = xb.to(self.device)
                self.opt.zero_grad()
                logits = self.decoder(self.encoder(xb))
                loss = self.loss_fn(logits, xb)
                loss.backward()
                self.opt.step()
                loss_sum += float(loss.item())
                n_batches += 1

            train_loss = loss_sum / max(1, n_batches)
            holdout_loss = self.reconstruction_loss(holdout_states)
            history.append({"epoch": ep + 1, "train_loss": train_loss, "holdout_loss": holdout_loss})
            print(f"[ae] epoch {ep + 1}/{self.epochs} train_loss={train_loss:.4f} holdout_loss={holdout_loss:.4f}")

        return history

    def encode(self, states: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.encoder.eval()
        out = []
        with torch.no_grad():
            for (xb,) in self._loader(states, shuffle=False):
                xb = xb.to(self.device)
                out.append(self.encoder(xb).cpu().numpy())
        return np.concatenate(out, axis=0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "in_channels": self.in_channels,
                "hidden": self.hidden,
                "latent": self.latent,
                "board_size": self.board_size,
            },
            path,
        )


def load_autoencoder(path: str, device: str = "cpu") -> ConvAutoencoder:
    """Rebuild a `ConvAutoencoder` saved by `ConvAutoencoder.save`, ready for
    `encode`/`reconstruction_loss` (eval mode, no optimiser state)."""
    import torch

    blob = torch.load(path, map_location=device)
    model = ConvAutoencoder(
        in_channels=blob["in_channels"],
        hidden=blob["hidden"],
        latent=blob["latent"],
        board_size=blob["board_size"],
        device=device,
    )
    model.encoder.load_state_dict(blob["encoder"])
    model.decoder.load_state_dict(blob["decoder"])
    model.encoder.eval()
    model.decoder.eval()
    return model
