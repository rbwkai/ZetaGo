"""PyTorch-based CNN policy/value model used as an optional baseline.

This module wraps a small convnet and provides `fit` and `predict` methods so
it can be used interchangeably with NumPy baselines. PyTorch is optional and
only required when the `cnn` model is selected.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..losses import LossBase, WeightedPolicyValueLoss
from ..metrics import softmax_np
from .base import SupervisedModel


class CNNModel(SupervisedModel):
    name = "cnn"

    @property
    def expects_flattened(self) -> bool:
        return False

    def __init__(
        self,
        in_channels: int,
        board_size: int = 7,
        num_moves: int = 50,
        hidden: int = 64,
        epochs: int = 12,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        loss_fn: LossBase | None = None,
    ):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
        except Exception as exc:
            raise RuntimeError("PyTorch is required for CNNModel") from exc

        self.torch = torch
        self.nn = nn
        self.DataLoader = DataLoader
        self.Dataset = Dataset

        self.board_size = board_size
        self.num_moves = num_moves
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed
        self.loss_fn = loss_fn or WeightedPolicyValueLoss(0.5)

        self.model = self._build_model(in_channels, hidden).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _build_model(self, in_channels: int, hidden: int):
        nn = self.nn
        b = self.board_size
        m = self.num_moves

        class PolicyValueNet(nn.Module):
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
                self.policy_head = nn.Sequential(
                    nn.Conv2d(hidden, 32, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(32 * b * b, m),
                )
                self.value_head = nn.Sequential(
                    nn.Conv2d(hidden, 16, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * b * b, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                    nn.Tanh(),
                )

            def forward(self, x):
                z = self.body(x)
                return self.policy_head(z), self.value_head(z).squeeze(1)

        return PolicyValueNet()

    def _make_dataset(self, x: np.ndarray, y_move: np.ndarray, y_value: np.ndarray):
        torch = self.torch

        class TensorSplit(self.Dataset):
            def __init__(self):
                self.x = torch.from_numpy(x.astype(np.float32))
                self.y_move = torch.from_numpy(y_move.astype(np.int64))
                self.y_value = torch.from_numpy(y_value.astype(np.float32))

            def __len__(self):
                return len(self.y_move)

            def __getitem__(self, idx):
                return self.x[idx], self.y_move[idx], self.y_value[idx]

        return TensorSplit()

    def fit(self, x_train: np.ndarray, y_move_train: np.ndarray, y_value_train: np.ndarray) -> None:
        torch = self.torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        ds = self._make_dataset(x_train, y_move_train, y_value_train)
        dl = self.DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model.train()
        for ep in range(self.epochs):
            loss_sum = 0.0
            n_batches = 0
            for xb, ymb, yvb in dl:
                xb = xb.to(self.device)
                ymb = ymb.to(self.device)
                yvb = yvb.to(self.device)

                self.opt.zero_grad()
                p_logits, v_pred = self.model(xb)
                loss = self.loss_fn(p_logits, v_pred, ymb, yvb)
                loss.backward()
                self.opt.step()

                loss_sum += float(loss.item())
                n_batches += 1

            avg_loss = loss_sum / max(1, n_batches)
            print(f"[cnn] epoch {ep + 1}/{self.epochs} loss={avg_loss:.4f}")

    def predict(self, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ds = self._make_dataset(x_val, np.zeros(len(x_val), dtype=np.int64), np.zeros(len(x_val), dtype=np.float32))
        dl = self.DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        all_move_logits = []
        all_value_pred = []
        with self.torch.no_grad():
            for xb, _, _ in dl:
                xb = xb.to(self.device)
                p_logits, v_pred = self.model(xb)
                all_move_logits.append(p_logits.cpu().numpy())
                all_value_pred.append(v_pred.cpu().numpy())

        move_logits = np.concatenate(all_move_logits, axis=0)
        move_proba = softmax_np(move_logits)
        value_pred = np.concatenate(all_value_pred, axis=0).astype(np.float32)
        return move_proba, value_pred
