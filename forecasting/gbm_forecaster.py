"""Gradient Boosting forecaster: lag + calendar features, tree ensemble.

Chosen as the GBM-based / feature-engineered ML paradigm (see the
literature review at the top of notebooks/02_experiments.ipynb): a
computationally cheap alternative to a sequence model when the modeler can
hand-engineer informative lags, competitive with deep sequence models on
network traffic while training in seconds (Kim, 2024/25).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from forecasting.base import BaseForecaster

# 10-30 min, 1-4 hours, and one full day - see 01_eda.ipynb's PACF result
# (direct dependence concentrated in the first ~5 lags) and ACF result
# (daily periodicity) for the justification.
LAGS = [1, 2, 3, 6, 12, 18, 24, 144]


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_weekend": (dow >= 5).astype("float32"),
        },
        index=index,
    )


def _feature_row(history: pd.Series, target_ts: pd.Timestamp) -> pd.DataFrame:
    """One feature row for predicting `target_ts`, built from lags of the
    known `history` (which must end at target_ts - one interval) plus
    calendar features of target_ts itself (always known in advance)."""
    row = {f"lag_{lag}": history.iloc[-lag] for lag in LAGS}
    row_df = pd.DataFrame([row], index=[target_ts])
    return row_df.join(_calendar_features(pd.DatetimeIndex([target_ts])))


class GBMForecaster(BaseForecaster):
    # A curated list of explicit combinations (not a dict-of-lists to be
    # cross-producted) - keeps the search space small and deliberate rather
    # than exhaustive. See HyperparameterSearch, which iterates PARAM_GRID
    # directly as one dict of kwargs per trial.
    PARAM_GRID = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: GradientBoostingRegressor | None = None

    def fit(self, series: pd.Series) -> "GBMForecaster":
        lag_data = {f"lag_{lag}": series.shift(lag) for lag in LAGS}
        table = pd.DataFrame(lag_data, index=series.index)
        table = table.join(_calendar_features(series.index))
        table["target"] = series.values
        table = table.dropna()

        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=0,
        )
        self.model.fit(table.drop(columns=["target"]).values, table["target"].values)
        return self

    def predict_one_step(self, history: pd.Series) -> float:
        if self.model is None:
            raise RuntimeError("call fit() before predict_one_step()")
        target_ts = history.index[-1] + (history.index.freq or pd.Timedelta(minutes=10))
        row = _feature_row(history, target_ts)
        return float(self.model.predict(row.values)[0])

    def describe(self) -> dict:
        return {
            "name": "Gradient Boosting (lag + calendar features)",
            "structure": (
                "scikit-learn GradientBoostingRegressor: an additive ensemble of "
                f"{self.n_estimators} shallow regression trees (max_depth="
                f"{self.max_depth}), each fit to the residuals of the ensemble so far, "
                f"combined with learning_rate={self.learning_rate}."
            ),
            "input_representation": (
                f"Lag features at offsets {LAGS} (10-30 min, 1-4 hours, and one full "
                "day) plus calendar features (sin/cos hour-of-day, sin/cos "
                "day-of-week, is_weekend) for the timestamp being predicted."
            ),
            "preprocessing": (
                "None beyond lag/calendar feature construction - tree split points "
                "are invariant to monotone feature scaling, unlike the LSTM."
            ),
            "training_procedure": (
                "Single batch fit on the training window's lag/calendar table; no "
                "iterative epochs, no validation-based early stopping within fit()."
            ),
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
        }
