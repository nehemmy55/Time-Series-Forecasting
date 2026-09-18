"""LSTM forecaster: a small end-to-end recurrent network (PyTorch, CPU).

Chosen as the RNN-family paradigm (see the literature review at the top of
notebooks/02_experiments.ipynb): learns temporal dependencies end-to-end
from a raw window of history, rather than from hand-picked lags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting.base import BaseForecaster

SEQ_LEN = 144  # one full day of 10-minute history
MAX_EPOCHS = 25
PATIENCE = 4
BATCH_SIZE = 256

torch.manual_seed(0)
torch.set_num_threads(4)


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    # A curated list of explicit combinations, not a full cartesian
    # product - see notebooks/02_experiments.ipynb for the grid-search log.
    PARAM_GRID = [
        {"hidden_size": 16, "num_layers": 1, "lr": 1e-3},
        {"hidden_size": 32, "num_layers": 1, "lr": 1e-3},
        {"hidden_size": 32, "num_layers": 2, "lr": 5e-4},
    ]

    def __init__(self, hidden_size: int = 32, num_layers: int = 1, lr: float = 1e-3):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.model: _LSTMNet | None = None
        self.mean_: float | None = None
        self.std_: float | None = None
        self.epochs_run_: int | None = None

    def fit(self, series: pd.Series) -> "LSTMForecaster":
        self.mean_, self.std_ = float(series.mean()), float(series.std())
        norm = ((series - self.mean_) / self.std_).values.astype("float32")

        n = len(norm) - SEQ_LEN
        X = np.lib.stride_tricks.sliding_window_view(norm, SEQ_LEN)[:n].astype("float32")
        y = norm[SEQ_LEN : SEQ_LEN + n].astype("float32")

        n_val = max(1, int(len(X) * 0.1))
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        self.model = _LSTMNet(self.hidden_size, self.num_layers)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(X_tr).unsqueeze(-1)
        ytr_t = torch.from_numpy(y_tr)
        Xval_t = torch.from_numpy(X_val).unsqueeze(-1)
        yval_t = torch.from_numpy(y_val)

        best_val, best_state, bad_epochs = float("inf"), None, 0
        n_train = len(Xtr_t)
        epoch = 0
        for epoch in range(MAX_EPOCHS):
            self.model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, BATCH_SIZE):
                idx = perm[start : start + BATCH_SIZE]
                opt.zero_grad()
                loss = loss_fn(self.model(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(Xval_t), yval_t).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.epochs_run_ = epoch + 1
        return self

    def predict_one_step(self, history: pd.Series) -> float:
        if self.model is None:
            raise RuntimeError("call fit() before predict_one_step()")
        window = ((history.iloc[-SEQ_LEN:] - self.mean_) / self.std_).values.astype("float32")
        x = torch.from_numpy(window).unsqueeze(0).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).item()
        return pred_norm * self.std_ + self.mean_

    def describe(self) -> dict:
        return {
            "name": "LSTM",
            "structure": (
                f"PyTorch LSTM, {self.num_layers} layer(s), hidden_size="
                f"{self.hidden_size}, followed by a linear head predicting the next "
                "value from the final hidden state."
            ),
            "input_representation": (
                f"A raw sliding window of the last {SEQ_LEN} true values (one full "
                "day) - unlike the tree model's hand-picked lags, the network learns "
                "for itself which points in the window matter."
            ),
            "preprocessing": (
                "Z-score normalization using mean/std computed once on the training "
                "window only (never re-fit on validation/test data)."
            ),
            "training_procedure": (
                f"Adam (lr={self.lr}) + MSE loss, mini-batches of {BATCH_SIZE}, up to "
                f"{MAX_EPOCHS} epochs with early stopping (patience={PATIENCE}) on an "
                "internal validation slice of the training window itself - a "
                f"stopping criterion only. Ran {self.epochs_run_} epoch(s) this fit."
            ),
            "hyperparameters": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "lr": self.lr,
            },
        }
