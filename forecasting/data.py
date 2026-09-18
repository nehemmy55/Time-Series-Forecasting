"""Memory-efficient data loading: two-pass raw-file aggregation and the
single reconstruction path every notebook uses to get one square's series.
"""
from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

RAW_COLUMNS = [
    "square_id", "timestamp_ms", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet_traffic",
]
USE_COLUMNS = [0, 1, 7]
USE_NAMES = ["square_id", "timestamp_ms", "internet_traffic"]
N_SQUARES = 10_000  # square ids run 1..10000 (dataset schema)

TUNING_TRAIN_END = pd.Timestamp("2013-12-08 23:50:00")
TUNING_VAL_END = pd.Timestamp("2013-12-15 23:50:00")
FINAL_TRAIN_END = TUNING_VAL_END
TEST_START = pd.Timestamp("2013-12-16 00:00:00")
TEST_END = pd.Timestamp("2013-12-22 23:50:00")
DAILY_PERIOD = 144  # 10-minute bins per day


class DataLoader:
    """Two-pass chunked aggregation of one raw daily file (or all of them)."""

    def __init__(self, chunksize: int = 1_000_000):
        self.chunksize = chunksize

    def _scan_timestamps(self, raw_path: Path) -> np.ndarray:
        seen = set()
        reader = pd.read_csv(
            raw_path, sep="\t", header=None, usecols=[1], names=["timestamp_ms"],
            dtype={"timestamp_ms": "int64"}, chunksize=self.chunksize,
        )
        for chunk in reader:
            seen.update(chunk["timestamp_ms"].unique().tolist())
        return np.array(sorted(seen), dtype="int64")

    def process_day(self, raw_path: Path, out_path: Path) -> dict:
        start = time.time()
        raw_path, out_path = Path(raw_path), Path(out_path)

        # Pass 1: which timestamps exist in this file?
        timestamps_ms = self._scan_timestamps(raw_path)
        ts_to_col = {ts: i for i, ts in enumerate(timestamps_ms)}
        n_ts = len(timestamps_ms)

        # Pass 2: accumulate traffic into a dense (square, timestamp) grid.
        matrix = np.zeros((N_SQUARES, n_ts), dtype="float32")
        n_rows_read = 0
        reader = pd.read_csv(
            raw_path, sep="\t", header=None, usecols=USE_COLUMNS, names=USE_NAMES,
            dtype={"square_id": "int16", "timestamp_ms": "int64", "internet_traffic": "float32"},
            chunksize=self.chunksize,
        )
        for chunk in reader:
            n_rows_read += len(chunk)
            chunk["internet_traffic"] = chunk["internet_traffic"].fillna(0.0)
            row_idx = (chunk["square_id"].to_numpy() - 1).astype("int64")
            col_idx = chunk["timestamp_ms"].map(ts_to_col).to_numpy(dtype="int64")
            np.add.at(matrix, (row_idx, col_idx), chunk["internet_traffic"].to_numpy())

        square_ids = np.repeat(np.arange(1, N_SQUARES + 1, dtype="int16"), n_ts)
        timestamps = np.tile(timestamps_ms, N_SQUARES)
        daily = pd.DataFrame({
            "square_id": square_ids,
            "timestamp_ms": timestamps,
            "internet_traffic": matrix.ravel(),
        })
        daily["timestamp"] = (
            pd.to_datetime(daily["timestamp_ms"], unit="ms", utc=True)
            .dt.tz_convert("Europe/Rome")
            .dt.tz_localize(None)
        )
        daily = daily.drop(columns=["timestamp_ms"]).sort_values(["square_id", "timestamp"])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(out_path, index=False, compression="snappy")

        return {
            "file": raw_path.name,
            "rows_read": n_rows_read,
            "rows_aggregated": len(daily),
            "seconds": round(time.time() - start, 1),
        }

    def build_all(self, raw_dir: Path, out_dir: Path) -> list[dict]:
        raw_dir, out_dir = Path(raw_dir), Path(out_dir)
        stats = []
        for raw_path in sorted(raw_dir.glob("sms-call-internet-mi-*.txt")):
            out_path = out_dir / (raw_path.stem + ".parquet")
            if out_path.exists():
                continue
            stats.append(self.process_day(raw_path, out_path))
        return stats

    def combine(self, daily_dir: Path, out_path: Path) -> pd.DataFrame:
        daily_dir, out_path = Path(daily_dir), Path(out_path)
        frames = [pd.read_parquet(p) for p in sorted(daily_dir.glob("*.parquet"))]
        combined = pd.concat(frames, ignore_index=True).sort_values(["square_id", "timestamp"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False, compression="snappy")
        return combined


def naive_load_day(raw_path: Path) -> int:
    """Baseline for the memory comparison: one full read_csv, no chunking or downcasting."""
    df = pd.read_csv(raw_path, sep="\t", header=None, names=RAW_COLUMNS)
    return len(df)


def _measure_worker(func, args, kwargs, queue) -> None:
    func(*args, **kwargs)
    mi = psutil.Process().memory_info()
    queue.put(int(getattr(mi, "peak_wset", mi.rss)))


def measure_peak_memory(func, *args, **kwargs) -> int:
    """Run `func` in an isolated child process and return its peak working-set size in bytes."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_measure_worker, args=(func, args, kwargs, queue))
    proc.start()
    peak = queue.get()
    proc.join()
    return peak


class SquareSeries:
    """The one path every notebook uses to reconstruct a square's traffic series."""

    def __init__(self, square_id: int, processed_path: Path):
        self.square_id = square_id
        self.processed_path = Path(processed_path)
        self._series: pd.Series | None = None

    def load(self) -> pd.Series:
        if self._series is None:
            df = pd.read_parquet(
                self.processed_path, filters=[("square_id", "==", self.square_id)]
            )
            s = (
                df.drop_duplicates(subset="timestamp")
                .set_index("timestamp")["internet_traffic"]
                .sort_index()
                .asfreq("10min")
                .interpolate()
                .astype("float32")
            )
            self._series = s
        return self._series
