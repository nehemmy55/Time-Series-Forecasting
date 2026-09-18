"""Forecasting package; model subclasses are lazy-imported below so plain
`import forecasting` doesn't pull in torch/statsmodels/sklearn."""
from forecasting.base import BaseForecaster
from forecasting.data import DataLoader, SquareSeries
from forecasting.evaluation import WalkForwardEvaluator, mae, mape, rmse
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

_LAZY = {
    "GBMForecaster": "forecasting.gbm_forecaster",
    "LSTMForecaster": "forecasting.lstm_forecaster",
    "SARIMAForecaster": "forecasting.sarima_forecaster",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
