# RGDGL

Implementation of **Reliability-Gated Dynamic Graph Learning for Missing-Sensor Traffic Imputation and Probabilistic Forecasting**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Obtain the METR-LA or PEMS-BAY speed data and sensor-distance tables from the DCRNN dataset distribution. The input HDF must contain a timestamp-indexed DataFrame with one sensor per column. A distance CSV has the columns `from,to,distance`; its IDs must match the HDF column names. Unlisted edges have infinite distance. Distances use one common nonnegative unit.

```bash
rgdgl prepare --input data/metr-la.h5 --distance data/distances.csv --output data/metr --realizations 10 --seed 13
```

An alternative NPZ input contains `values[time,node,channel]`, `distance[node,node]` and optional `clock[time]` as fractions of a day. A two-dimensional value array becomes one channel. When clock is absent, samples are interpreted as consecutive five-minute intervals beginning at midnight.

Finite readings are available, including numerical zeros. Declare `--sentinel VALUE` only when the dataset metadata identifies that value as unavailable. NaN/Inf and the explicit sentinel are excluded from targets before normalization. Channel statistics use available training readings only. The chronological split is 70/10/20 and windows never cross it.

Preparation saves the normalized train/validation/test timelines, source-availability masks, distance matrix and a fixed bank of global fault masks. Each family has rates 0, 0.1, 0.3 and 0.5 and ten realizations by default. All methods consume these same files.

## Training and inference

```bash
rgdgl train --data data/metr --output outputs/full --variant full --seed 13
```

## Evaluation

```bash
rgdgl evaluate --data data/metr --model outputs/full/model.pt --family random --rate 0.3 --realization 0 --output outputs/full_random03
rgdgl plot --metrics outputs/full_random03/metrics.json outputs/nogate_random03/metrics.json --output outputs/figures
```

Keep fault family, rate and realization identical between checkpoints.

## Comparisons and ablations

```bash
rgdgl train --data data/metr --output outputs/static --variant static --seed 13
rgdgl train --data data/metr --output outputs/no_gate --variant no_gate --seed 13
rgdgl train --data data/metr --output outputs/no_correlation --variant no_correlation --seed 13
rgdgl train --data data/metr --output outputs/no_imputation --variant no_imputation --seed 13
rgdgl evaluate --data data/metr --model outputs/no_gate/model.pt --family random --rate 0.3 --realization 0 --output outputs/nogate_random03
```

## Outputs

Training writes `model.pt` and `history.json`. Evaluation writes `metrics.json` and `predictions.npz` with forecasts, variances, targets and availability.

```bash
rgdgl plot --metrics outputs/full_random03/metrics.json outputs/nogate_random03/metrics.json --output outputs/figures
rgdgl-compare --first outputs/full_random03/predictions.npz --second outputs/nogate_random03/predictions.npz --output outputs/paired.json --block 288 --resamples 10000
```
