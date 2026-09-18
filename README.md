# Mobile Network Traffic Forecasting

Comparative analysis of sequential models for one-step-ahead mobile network
traffic forecasting, using the Telecom Italia "Milano Grid" SMS-Call-Internet
dataset (Nov 1, 2013 - Jan 1, 2014; 10,000 geographical areas; 10-minute
intervals).

The project is notebook-first: all narrative analysis and results live in
`notebooks/`. `forecasting/` is a small class-based package the notebooks
call into, so every model and every evaluation goes through the same
interface (`BaseForecaster` -> `WalkForwardEvaluator`) by construction rather
than each notebook reimplementing its own prediction loop.

## Package organization (`forecasting/`)

| Module | Contents |
|---|---|
| `base.py` | `BaseForecaster` - the abstract interface every model implements: `fit(series, exog=None)`, `predict_one_step(history) -> float`, `describe() -> dict`, and a `PARAM_GRID` class attribute. |
| `data.py` | `DataLoader` (two-pass chunked aggregation of raw daily files), `SquareSeries` (the one path used everywhere to load/resample/interpolate a square's series), `naive_load_day` / `measure_peak_memory` (the Section-1 memory comparison), and the standard train/val/test split boundaries. |
| `sarima_forecaster.py` | `SARIMAForecaster` - Fourier-augmented SARIMA (statsmodels). |
| `gbm_forecaster.py` | `GBMForecaster` - gradient boosting over lag + calendar features (scikit-learn). |
| `lstm_forecaster.py` | `LSTMForecaster` - a small LSTM (PyTorch, CPU). |
| `evaluation.py` | `WalkForwardEvaluator` and MAE/MAPE/RMSE - the *only* place evaluation logic lives. True one-step-ahead: at each timestamp the model conditions on real history only, never its own prior prediction. |
| `tracking.py` | `ExperimentTracker` - appends every training run (model, params, metrics, timestamp, a one-line rationale) to `results/experiment_log.csv`. |
| `search.py` | `HyperparameterSearch` - runs a `BaseForecaster` subclass's `PARAM_GRID` through `WalkForwardEvaluator` on a validation window, logging every trial plus a closing rationale for the winner via `ExperimentTracker`. |
| `viz.py` | Shared matplotlib style (colorblind-safe palette) used by every figure. |
| `utils.py` | `hardware_info()` - used to report the machine the timing numbers were measured on. |

## Project structure

```
forecasting/            the package described above
data/
  raw/                   62 daily .txt files (gitignored, ~20GB total)
  processed/
    daily/*.parquet      one aggregated file per raw day (DataLoader output)
    internet_traffic.parquet   combined dataset: square_id, timestamp, internet_traffic
notebooks/
  00_data_pipeline.ipynb      Section 1: raw-file check, memory comparison, build the dataset, rank squares
  01_eda.ipynb                Section 2: distribution, 5-square series, periodicity heatmap, ACF/PACF+ADF
  02_experiments.ipynb        Section 3+4: literature review, hyperparameter search, final evaluation
  03_model_comparison.ipynb   Section 4 outputs: 9 plots, 3 tables, timing, failure case
figures/                 all figures saved by the notebooks
results/                 experiment_log.csv, top_squares.json, per-square metrics/timing CSVs, saved predictions
report/                  (reserved for the final PDF report - not built yet)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
# torch is CPU-only; if the default index pulls a CUDA build, use:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Register a Jupyter kernel for this venv (used by `jupyter nbconvert --execute`
# and by the "mobiletraffic (.venv)" kernel the notebooks are saved with):
python -m ipykernel install --user --name=mobiletraffic --display-name "mobiletraffic (.venv)"
```

Download the raw daily `.txt` files (Telecom Italia SMS-Call-Internet-MI
dataset, [1][2]) into `data/raw/`.

## Running the notebooks

Run in order - each one reads files the previous one wrote to `data/processed/`
or `results/`:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mobiletraffic 00_data_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mobiletraffic 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mobiletraffic 02_experiments.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mobiletraffic 03_model_comparison.ipynb
```

Or open them in Jupyter/VS Code and run all cells top to bottom in the same
order. `00_data_pipeline.ipynb` is the slow one (~15-20 min - two passes over
each of the 62 raw files); `02_experiments.ipynb` is the next slowest
(~10-15 min - hyperparameter search + 9 final model/square fits, LSTM
training dominates).

## Adding a new model

1. Subclass `forecasting.base.BaseForecaster` in a new
   `forecasting/<name>_forecaster.py`.
2. Implement `fit(series, exog=None)` (return `self`), `predict_one_step
   (history) -> float`, and `describe() -> dict` (name/structure/input
   representation/preprocessing/training procedure/hyperparameters - phrased
   so it can be pasted into a report's Methodology section).
3. Set a `PARAM_GRID` class attribute: a list of `dict`s, one per
   hyperparameter combination `HyperparameterSearch` should try (not a
   dict-of-lists to be cross-producted - keep the search space curated and
   small, matching the SARIMA/GBM/LSTM examples).
4. If `predict_one_step` would be expensive to call once per evaluated
   timestamp when called via `WalkForwardEvaluator` (as SARIMA's Kalman
   filter update is), cache whatever internal state lets repeated calls stay
   cheap - see `SARIMAForecaster` for the pattern (track how much of
   `history` has already been incorporated, append only the new tail).
5. Nothing else needs to change: `HyperparameterSearch`, `WalkForwardEvaluator`,
   and `ExperimentTracker` all work against the `BaseForecaster` interface,
   and `02_experiments.ipynb` only needs the new class added to its
   `forecaster_classes` dict to include it in the comparison.

## Data handling and memory management

The raw dataset is ~20GB across 62 daily files (~4.8M rows/day: one row per
`square_id x 10-minute interval x country_code`). `DataLoader.process_day`
(`forecasting/data.py`) aggregates one day in two passes:

1. **Pass 1** streams the file in chunks, reading only the timestamp column,
   to collect the sorted set of unique 10-minute timestamps present (~144).
2. **Pass 2** re-streams the file (square_id, timestamp, internet_traffic
   only) and accumulates directly into a preallocated
   `(10,000 squares x ~144 timestamps)` `float32` array via `np.add.at`
   (correctly sums the multiple country-code rows per square/timestamp),
   instead of building and concatenating per-chunk partial DataFrames.

This bounds peak memory by the array size (a few MB) plus one chunk's worth
of raw rows - independent of file size or chunk count, an improvement on a
single-pass chunk-then-concat-then-groupby strategy whose intermediate memory
scales somewhat with chunk count. `00_data_pipeline.ipynb` measures this
against a naive single-`read_csv`-all-columns baseline and reports the
before/after numbers with hardware details.

**Trade-offs.** The two-pass approach reads each raw file's timestamp column
twice, trading a small amount of extra I/O and bookkeeping code for a memory
bound that no longer depends on file size. The output is also a *dense*
square x timestamp grid (absent combinations become an explicit 0.0 rather
than a missing row) - a deliberate choice that guarantees a complete regular
grid without a later per-day resample step, at the cost of a small, fixed
amount of extra output size (see `forecasting/data.py`'s `DataLoader`
docstring).

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city
of Milan and the Province of Trentino," *Sci Data*, vol. 2, 150055, 2015.
[2] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI,"
Harvard Dataverse, doi:10.7910/DVN/EGZHFV.

The full literature review behind the three model choices (5 sources, IEEE
style) is in the markdown cell at the top of `notebooks/02_experiments.ipynb`.
