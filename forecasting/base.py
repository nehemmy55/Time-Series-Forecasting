"""Common interface every forecasting model implements."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Abstract one-step-ahead forecaster; subclasses set PARAM_GRID for HyperparameterSearch."""

    PARAM_GRID: list = []

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseForecaster":
        """Fit the model on a training series. Returns self (chaining)."""
        raise NotImplementedError

    @abstractmethod
    def predict_one_step(self, history: pd.Series) -> float:
        """Predict the value immediately following `history`."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        """Self-description for the report's Methodology section."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.describe().get('hyperparameters', {})})"
