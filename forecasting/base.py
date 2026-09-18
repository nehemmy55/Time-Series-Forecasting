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
    declare a `PARAM_GRID` class attribute (a curated list of kwargs dicts,
    one per trial) describing the space HyperparameterSearch should
    explore.
    """

    #: list[dict] — overridden by each subclass.
    PARAM_GRID: list = []

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseForecaster":
        """Fit the model on a training series. Returns self (chaining).

        Deliberately no `exog` parameter: each of the three concrete models
        needs a different, model-specific derived signal (SARIMA's Fourier
        terms, none for GBM/LSTM beyond what's already in `series`), and
        all three compute it internally from `series.index` rather than
        accepting it from the caller - so a generic external-exog parameter
        would be accepted by the interface but ignored by every subclass.
        Add it back only if a future model genuinely needs caller-supplied
        exogenous data it cannot derive from the series' own index.
        """
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
