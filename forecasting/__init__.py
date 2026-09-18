"""Small class-based forecasting package backing the project's notebooks.

See README.md for package organization and how to add a new model.

The three model subclasses (GBMForecaster, LSTMForecaster, SARIMAForecaster)
are imported lazily (PEP 562 module __getattr__) rather than eagerly here,
so that `import forecasting` - or, critically, unpickling a plain function
like `DataLoader.process_day` in a multiprocessing child process, which
Python does by re-importing its defining module and therefore this package
- doesn't transitively pull in torch/statsmodels/sklearn. That confound
would otherwise inflate notebooks/00_data_pipeline.ipynb's naive-vs-optimized
memory comparison with irrelevant import overhead unrelated to either
loading strategy.
"""
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
