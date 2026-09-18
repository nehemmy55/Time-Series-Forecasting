"""SARIMA-family statistical forecaster: Fourier-augmented ARIMA.

Chosen as the statistical paradigm (see the literature review at the top
of notebooks/02_experiments.ipynb). A *literal* seasonal ARIMA term at the
daily period (statsmodels `seasonal_order=(P,D,Q,144)`) is impractical on
laptop-class hardware - a single such fit was measured taking over 19
CPU-minutes without converging, versus ~5 seconds for a comparably-sized
plain ARIMA fit (see notebooks/02_experiments.ipynb). Instead this follows
the standard remedy for long seasonal periods (Hyndman & Athanasopoulos,
"Forecasting: Principles and Practice", ch. 12): Fourier sine/cosine
exogenous regressors for the daily cycle, plus a weekend dummy (see
01_eda.ipynb's periodicity heatmap), wrapped around a low-order ARIMA
error process.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.base import BaseForecaster

DAILY_PERIOD = 144


def _fourier_and_weekend(index: pd.DatetimeIndex, n_harmonics: int) -> np.ndarray:
    """Fourier phase computed from each timestamp's actual time-of-day
    (minutes since midnight / 10), NOT from position within `index`. This
    matters because predict_one_step() calls this on single-row slices one
    step at a time - a position-based `t = arange(len(index))` would reset
    to phase 0 (midnight) on every single call regardless of the row's real
    time of day, silently corrupting the seasonal signal. Computing phase
    from calendar time instead makes the result correct for any slice,
    contiguous or not, one row or many."""
    t = (index.hour * 60 + index.minute) / 10  # 10-minute bin within the day, 0..143
    cols = {}
    for k in range(1, n_harmonics + 1):
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / DAILY_PERIOD)
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / DAILY_PERIOD)
    cols["is_weekend"] = (index.dayofweek >= 5).astype("float64")
    return pd.DataFrame(cols, index=index).values


class SARIMAForecaster(BaseForecaster):
    # A curated list of explicit combinations, not a full cartesian
    # product - see notebooks/02_experiments.ipynb for the grid-search log.
    PARAM_GRID = [
        {"order": (1, 0, 1), "n_harmonics": 2},
        {"order": (1, 0, 1), "n_harmonics": 3},
        {"order": (2, 0, 1), "n_harmonics": 2},
        {"order": (2, 0, 1), "n_harmonics": 3},
        {"order": (2, 1, 2), "n_harmonics": 2},
        {"order": (2, 1, 2), "n_harmonics": 3},
    ]

    def __init__(self, order: tuple = (2, 1, 2), n_harmonics: int = 3):
        self.order = order
        self.n_harmonics = n_harmonics
        self._results = None
        self._incorporated_len = 0
        self._freq: pd.Timedelta | None = None

    def fit(self, series: pd.Series) -> "SARIMAForecaster":
        exog_arr = _fourier_and_weekend(series.index, self.n_harmonics)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series.values, exog=exog_arr, order=self.order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self._results = model.fit(disp=False, maxiter=100)
        self._incorporated_len = len(series)
        self._freq = series.index.freq or pd.Timedelta(minutes=10)
        self._last_index = series.index[-1]
        return self

    def predict_one_step(self, history: pd.Series) -> float:
        """One call per evaluated timestamp, but internally O(1) amortized:
        only the new tail of `history` since the last call is appended to
        the cached filtered state (refit=False, a cheap Kalman-filter
        update), not the whole history each time."""
        if self._results is None:
            raise RuntimeError("call fit() before predict_one_step()")

        if len(history) > self._incorporated_len:
            new_tail = history.iloc[self._incorporated_len :]
            new_exog = _fourier_and_weekend(new_tail.index, self.n_harmonics)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._results = self._results.append(new_tail.values, exog=new_exog, refit=False)
            self._incorporated_len = len(history)
            self._last_index = history.index[-1]

        next_ts = self._last_index + self._freq
        next_exog = _fourier_and_weekend(pd.DatetimeIndex([next_ts]), self.n_harmonics)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._results.forecast(steps=1, exog=next_exog)
        return float(forecast[0])

    def describe(self) -> dict:
        return {
            "name": "SARIMA (Fourier-augmented ARIMA)",
            "structure": (
                f"statsmodels SARIMAX, ARIMA order {self.order}, with "
                f"{self.n_harmonics} Fourier harmonic pairs (period {DAILY_PERIOD}, "
                "the daily cycle) plus a weekend dummy as exogenous regressors, in "
                "place of a literal seasonal_order term (see module docstring)."
            ),
            "input_representation": (
                "The full training series' levels, plus per-timestep exogenous "
                f"Fourier sin/cos pairs (k=1..{self.n_harmonics}) and an is_weekend "
                "dummy."
            ),
            "preprocessing": "None - SARIMAX estimates its own error-process structure.",
            "training_procedure": (
                "Single maximum-likelihood fit (state-space form) on the training "
                "window; at prediction time, new true observations are appended to "
                "the fitted filter (refit=False) rather than re-estimating "
                "parameters, keeping one-step-ahead walk-forward evaluation cheap."
            ),
            "hyperparameters": {"order": self.order, "n_harmonics": self.n_harmonics},
        }
