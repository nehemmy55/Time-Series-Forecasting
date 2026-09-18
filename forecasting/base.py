"""Common interface every forecasting model implements.

Every model in this project (SARIMAForecaster, GBMForecaster, LSTMForecaster)
subclasses BaseForecaster, which is the only reason WalkForwardEvaluator,
HyperparameterSearch, and the comparison notebook can treat all three
identically instead of special-casing each one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Abstract one-step-ahead forecaster.

    Subclasses store their hyperparameters on `self` in `__init__` and
    declare a `PARAM_GRID` class attribute (dict of {param_name: [values]})
    describing the space HyperparameterSearch should explore.
    """

    #: dict[str, list] — overridden by each subclass.
    PARAM_GRID: dict = {}

    @abstractmethod
    def fit(self, series: pd.Series, exog: pd.DataFrame | None = None) -> "BaseForecaster":
        """Fit the model on a training series. Returns self (chaining)."""
        raise NotImplementedError

    @abstractmethod
    def predict_one_step(self, history: pd.Series) -> float:
        """Predict the value immediately following `history`.

        `history` is the *true* series up to and including some time t;
        the return value is the model's estimate of x(t+1). Called once per
        evaluated timestamp by WalkForwardEvaluator - implementations that
        need repeated calls to be cheap (e.g. a state-space model) should
        cache internal state across calls rather than reprocessing all of
        `history` from scratch each time.
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        """Self-description for the report's Methodology section.

        Expected keys: name, structure, input_representation, preprocessing,
        training_procedure, hyperparameters (the instance's current values).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.describe().get('hyperparameters', {})})"
