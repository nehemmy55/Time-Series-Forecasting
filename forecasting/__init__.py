"""Small class-based forecasting package backing the project's notebooks.

See README.md for package organization and how to add a new model.
"""
from forecasting.base import BaseForecaster
from forecasting.data import DataLoader, SquareSeries
from forecasting.evaluation import WalkForwardEvaluator, mae, mape, rmse
from forecasting.gbm_forecaster import GBMForecaster
from forecasting.lstm_forecaster import LSTMForecaster
from forecasting.sarima_forecaster import SARIMAForecaster
from forecasting.search import HyperparameterSearch
from forecasting.tracking import ExperimentTracker
from forecasting.utils import hardware_info

__all__ = [
    "hardware_info",
    "BaseForecaster",
    "DataLoader",
    "SquareSeries",
    "WalkForwardEvaluator",
    "mae",
    "mape",
    "rmse",
    "GBMForecaster",
    "LSTMForecaster",
    "SARIMAForecaster",
    "HyperparameterSearch",
    "ExperimentTracker",
]
