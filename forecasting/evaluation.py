"""One-step-ahead walk-forward evaluation and the MAE/MAPE/RMSE metrics."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from forecasting.base import BaseForecaster


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    # eps guards against the near-zero overnight troughs blowing up the
    # percentage error.
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


class WalkForwardEvaluator:
    """At each target timestamp, conditions only on real history, never a prior prediction."""

    def run(self, model: BaseForecaster, full_series: pd.Series, eval_index: pd.DatetimeIndex) -> dict:
        freq = full_series.index.freq or pd.Timedelta(minutes=10)
        preds, actuals, kept = [], [], []

        start = time.perf_counter()
        for t in eval_index:
            history_end = t - freq
            if history_end not in full_series.index:
                continue  # not enough history to condition on (e.g. start of series)
            history = full_series.loc[:history_end]
            pred = model.predict_one_step(history)
            preds.append(pred)
            actuals.append(full_series.loc[t])
            kept.append(t)
        predict_seconds = time.perf_counter() - start

        kept_index = pd.DatetimeIndex(kept)
        pred_s = pd.Series(preds, index=kept_index)
        true_s = pd.Series(actuals, index=kept_index)

        return {
            "predictions": pred_s,
            "actuals": true_s,
            "mae": mae(true_s.values, pred_s.values),
            "mape": mape(true_s.values, pred_s.values),
            "rmse": rmse(true_s.values, pred_s.values),
            "n_steps": len(kept_index),
            "predict_seconds": predict_seconds,
        }
