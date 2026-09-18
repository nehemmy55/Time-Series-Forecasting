"""Structured experiment logging: every training run is recorded here."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = [
    "timestamp", "model", "square_id", "phase", "params",
    "mae", "mape", "rmse", "train_seconds", "predict_seconds", "rationale",
]


class ExperimentTracker:
    """Appends one row per run to `log_path` (CSV, created with a header on first use)."""

    def __init__(self, log_path: str | Path = "results/experiment_log.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        model: str,
        params: dict,
        metrics: dict,
        rationale: str,
        square_id: int | None = None,
        phase: str = "trial",
        train_seconds: float | None = None,
        predict_seconds: float | None = None,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "square_id": square_id,
            "phase": phase,
            "params": json.dumps(params),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "rmse": metrics.get("rmse"),
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds if predict_seconds is not None else metrics.get("predict_seconds"),
            "rationale": rationale,
        }
        row_df = pd.DataFrame([row], columns=COLUMNS)
        write_header = not self.log_path.exists()
        row_df.to_csv(self.log_path, mode="a", header=write_header, index=False)

    def read_log(self) -> pd.DataFrame:
        return pd.read_csv(self.log_path, parse_dates=["timestamp"])
