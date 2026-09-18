# Mobile Network Traffic Forecasting

Comparative analysis of sequential models for one-step-ahead mobile network
traffic forecasting, using the Telecom Italia "Milano Grid" SMS-Call-Internet
dataset (Nov 1, 2013 - Jan 1, 2014; 10,000 geographical areas; 10-minute
intervals).

## Project structure

```
data/
  sms-call-internet-mi-*.txt   raw daily files (gitignored, ~20GB total)
  processed/
    daily/*.parquet            one aggregated file per raw day
    internet_traffic.parquet   combined dataset: square_id, timestamp, internet_traffic
src/
  data/
    schema.py                  raw column layout + dtypes
    ingest.py                  chunked, memory-efficient aggregation of one raw day
    build_dataset.py           runs ingest.py over all raw days and combines them
    memory_naive.py            baseline memory profiling (single full read_csv)
    memory_optimized.py        optimized memory profiling (chunked + downcast)
  analysis/
    style.py                   shared plotting style (colorblind-safe palette)
    eda.py                     exploratory analysis, produces reports/figures/*.png
  models/
    common.py                  splits, input representation, metrics, timing, hardware info
    sarima_model.py            Model 1: Fourier-augmented SARIMA (statsmodels)
    gbm_model.py                Model 2: gradient boosting on lag + calendar features (sklearn)
    lstm_model.py               Model 3: LSTM (PyTorch, CPU)
    run_experiments.py         orchestrates tuning + evaluation + all Section 4 outputs
reports/
  figures/                     generated figures (EDA + 9 forecast plots + failure case)
  tables/                      per-square metrics CSVs + timing.csv
  eda_summary.txt              generated numeric evidence (top squares, ADF test, ...)
  experiment_log.md            hyperparameter search trials + hardware info
  report.md                    full research report (Sections 1-8)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
# torch is CPU-only; if the default index pulls a CUDA build, use:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Download the raw daily `.txt` files (Telecom Italia SMS-Call-Internet-MI
dataset, [1][2]) into `data/`.

## Running the pipeline

```bash
# 1. Aggregate raw text -> compact per-day Parquet, then combine
cd src/data
python build_dataset.py

# 2. Exploratory analysis (figures + eda_summary.txt)
cd ../analysis
python eda.py

# 3. Forecasting experiments: hyperparameter search + final evaluation for
#    all 3 models across the 3 highest-traffic squares (Section 4). Produces
#    9 forecast plots + a failure-case plot in reports/figures/, per-square
#    metrics tables in reports/tables/, and a full trial log with hardware
#    info in reports/experiment_log.md. Takes ~30-45 min on a 4-core CPU
#    laptop (LSTM training dominates the runtime).
cd ../models
python run_experiments.py
```

## Data handling and memory management

The raw dataset is ~20GB across 62 daily files (~4.8M rows/day: one row per
`square_id x 10-minute interval x country_code`). Loading a day naively with
`pandas.read_csv` (all 8 columns, default dtypes) peaks at **~453MB** of
working set for a single ~322MB file. Extrapolated across all 62 files at
once, a naive full-dataset load would require on the order of **25-30GB**
of RAM - impractical on a typical laptop.

The pipeline instead (`src/data/ingest.py`):
1. Reads each file in **1M-row chunks** instead of all at once, bounding
   peak memory to a small, constant multiple of the chunk size regardless
   of file size.
2. Reads only the **3 of 8 columns** needed for internet-traffic
   forecasting (`square_id`, `timestamp_ms`, `internet_traffic`), skipping
   SMS/call columns entirely at parse time.
3. **Downcasts dtypes** - `square_id` to `int16` (ids run 1-10,000),
   `internet_traffic` to `float32` - since the raw `int64`/`float64` range
   is never needed.
4. **Aggregates away the `country_code` dimension** per chunk (summing
   internet traffic per `square_id, timestamp`), then combines the
   per-chunk partial sums with a final groupby-sum. Summation is
   associative, so chunkwise partial aggregation is exact.
5. Persists one compact, compressed **Parquet** file per day instead of
   keeping raw text or full precision in memory.

Measured on the same file (`sms-call-internet-mi-2013-11-01.txt`, 322MB,
4,842,625 raw rows -> 1,439,982 aggregated rows), each approach run in its
own process (`src/data/memory_naive.py` / `memory_optimized.py`), peak
Windows working set (`psutil` `peak_wset`):

| Approach | Peak working set | Notes |
|---|---|---|
| Naive (`read_csv`, all columns, default dtypes) | ~453 MB | scales linearly with file size; does not fit the full 62-file dataset in memory at once |
| Chunked + column pruning + downcast + aggregation | ~282 MB | bounded by chunk size; independent of how many files are processed |

The full aggregated dataset (10,000 squares x ~144 intervals/day x 62 days)
compresses from **20GB raw text to a few hundred MB of Parquet**, and loads
back into memory comfortably (`int16`/`float32` columns) for the
exploratory analysis and modeling stages.

**Trade-offs.** Chunked aggregation trades a small amount of code
complexity (partial-sum combination) for a large, dataset-size-independent
memory bound. Downcasting to `float32` discards precision beyond ~7
significant digits, which is immaterial for traffic values of this
magnitude but would not be appropriate for computations requiring
higher numerical precision.

## Full report

See [`reports/report.md`](reports/report.md) for the complete research report
(introduction, related work, methodology, results, discussion, conclusion,
and full IEEE-style reference list).

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city
of Milan and the Province of Trentino," *Sci Data*, vol. 2, 150055, 2015.
[2] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI,"
Harvard Dataverse, doi:10.7910/DVN/EGZHFV.
