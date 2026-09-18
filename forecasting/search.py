"""Hyperparameter search over a BaseForecaster subclass's PARAM_GRID."""
from __future__ import annotations

import time

import pandas as pd

from forecasting.base import BaseForecaster
from forecasting.evaluation import WalkForwardEvaluator
from forecasting.tracking import ExperimentTracker


class HyperparameterSearch:
    def __init__(
        self,
        forecaster_cls: type[BaseForecaster],
        tracker: ExperimentTracker,
        param_grid: list[dict] | None = None,
    ):
        self.forecaster_cls = forecaster_cls
        self.param_grid = param_grid if param_grid is not None else forecaster_cls.PARAM_GRID
        self.tracker = tracker

    def run(
        self,
        full_series: pd.Series,
        train_end: pd.Timestamp,
        val_index: pd.DatetimeIndex,
        square_id: int | None = None,
        model_name: str | None = None,
    ) -> dict:
        model_name = model_name or self.forecaster_cls.__name__
        train_series = full_series.loc[:train_end]
        evaluator = WalkForwardEvaluator()

        all_results = []
        best = None
        for params in self.param_grid:
            model = self.forecaster_cls(**params)
            t0 = time.perf_counter()
            model.fit(train_series)
            train_seconds = time.perf_counter() - t0

            result = evaluator.run(model, full_series, val_index)
            metrics = {"mae": result["mae"], "mape": result["mape"], "rmse": result["rmse"]}

            self.tracker.log(
                model=model_name, params=params, metrics=metrics,
                rationale="grid search trial", square_id=square_id, phase="tuning",
                train_seconds=train_seconds, predict_seconds=result["predict_seconds"],
            )
            all_results.append({"params": params, "train_seconds": train_seconds, **metrics})
            if best is None or metrics["rmse"] < best["rmse"]:
                best = {"params": params, **metrics}

        rationale = (
            f"Selected {best['params']} - lowest validation RMSE "
            f"({best['rmse']:.2f}) among {len(self.param_grid)} candidates."
        )
        self.tracker.log(
            model=model_name, params=best["params"], metrics=best,
            rationale=rationale, square_id=square_id, phase="selected",
        )
        return {
            "best_params": best["params"],
            "best_metrics": {k: best[k] for k in ("mae", "mape", "rmse")},
            "all_results": all_results,
            "rationale": rationale,
        }
