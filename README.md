# RGDGL

Implementation of **Reliability-Gated Dynamic Graph Learning for Missing-Sensor Traffic Imputation and Probabilistic Forecasting**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0, pandas 2.2.3. CPU core verification: `python -m pytest -q tests/test_functional.py` passed 3 checks on 2026-11-01. A synthetic 160-step, 3-sensor timeline exercised preparation, one training epoch, validation-only variance fitting, checkpoint reload, forecast metrics and plots. All five variants passed finite loss/backward/parameter-update and exact eval round-trip checks; learned graph scalars received gradients in the full model. These checks are not paper performance results. Full METR-LA/PEMS-BAY experiments were not run.

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

## Method and metric entry points

| Paper component | Entry point | Command/configuration |
| --- | --- | --- |
| Causal correlation, reliability gate, graph GRU, dual heads | `rgdgl/model.py:ReliabilityGraph` | `rgdgl train --variant full` |
| Masked reconstruction + Gaussian NLL + entropy | `rgdgl/model.py:loss` | `--imputation-weight 1 --entropy-weight .001` |
| Chronological train-only scaling and global fault bank | `rgdgl/data.py:prepare` | `rgdgl prepare` |
| MAE/RMSE/MAPE, reconstruction MAE, NLL/CRPS and coverage/width | `rgdgl/cli.py:metrics` | `rgdgl evaluate` |
| Robustness slopes, paired block bootstrap, batch-one costs | `rgdgl/cli.py:main`, `rgdgl/analysis.py` | `rgdgl plot`, `rgdgl-compare`, `rgdgl-benchmark` |

This repository provides the graph method, five controls and native-source adapters for every external baseline family listed in the manuscript, as detailed below. The graph message uses dropout 0.1 (train mode only). Checkpoints use lexicographic validation (NLL, MAE). Intentional 10% clean windows retain a zero reconstruction mask; other empty training masks are resampled within the same fault family conditional on a nonempty eligible target. Completely unobserved input windows are resampled by excluding them from the training sampling pool. For finite windows whose rounded nonzero missingness budget is zero, this conditional sampler hides one eligible reading (or one eligible sensor for persistent faults). Validation/test banks remain fixed and unchanged. No private traffic data, weights or paper result arrays are included.


## Native baseline sources and common protocol

```bash
python -m pip install -e ".[baselines]"
python scripts/fetch_baselines.py
rgdgl train --data data/metr --forecaster agcrn --imputer forward --objective point --seed 13 --output outputs/forward_agcrn_point
rgdgl train --data data/metr --forecaster agcrn --imputer saits --objective gaussian --seed 13 --output outputs/saits_agcrn_gaussian
rgdgl train --data data/metr --forecaster gcnm --objective gaussian --seed 13 --output outputs/gcnm_gaussian
rgdgl train --data data/metr --forecaster uignn --objective evidential --seed 13 --output outputs/uignn_native
rgdgl evaluate --data data/metr --model outputs/gcnm_gaussian/model.pt --family mixed --rate .5 --realization 0 --output outputs/gcnm_mixed05
```

`baseline_sources.json` records the exact source revisions and inspected reuse licenses. The fetcher downloads sparse source checkouts into ignored `.external/`, never downloads traffic datasets, and leaves an existing pinned checkout intact. Upstream source is not copied into this repository. UIGNN and GRIN have no top-level reuse license in their inspected revisions; their runtime checkouts are therefore required separately and are not redistributed. The adapters execute their actual original definitions.

| Forecaster/imputer | Source and retained mechanism |
| --- | --- |
| `dcrnn` | [DCRNN PyTorch port](https://github.com/chnsh/DCRNN_PyTorch): bidirectional diffusion-GRU encoder/decoder; this is identified as the PyTorch port rather than the authors' TensorFlow runtime |
| `graphwavenet` | [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet): dilated gated temporal convolutions, skip/residual paths and learned adaptive adjacency |
| `agcrn`, `agcrn_mask` | [AGCRN](https://github.com/LeiBAI/AGCRN): node-adaptive parameter pools/graph recurrent cells; the mask control adds the actual observation channel |
| `brits` + AGCRN | BRITS recurrent temporal/feature regression and forward/backward consistency from the [GRIN reference implementation](https://github.com/Graph-Machine-Learning-Group/grin) |
| `saits` + AGCRN | [SAITS](https://github.com/WenjieDu/SAITS): both diagonal-masked self-attention blocks, attention-weighted combination, observed reconstruction and hidden-target losses |
| `grin` + AGCRN | [GRIN](https://github.com/Graph-Machine-Learning-Group/grin): bidirectional graph recurrent imputation and spatial decoding |
| `maginet` + AGCRN | [MagiNet](https://github.com/JeremyChou28/MagiNet): original adaptive missing encoder and complete mask-aware spatio-temporal decoder |
| `gcnm` | [GCN-M](https://github.com/JingweiZuo/GCN-M): original eight-feature local module, attention historical memory and dynamic graph forecaster |
| `uignn` | [UIGNN](https://github.com/lijunxian111/UIGNN): original inductive diffusion network, recovery objective and optional NIG evidential head/loss |

`mean`, `forward`, and five-neighbor `knn` pair with DCRNN, Graph WaveNet or AGCRN. Means/KNN reference records are fitted on source training only; forward fill uses observations at or before the input step, and KNN receives only the prefix ending at the forecast origin. Dedicated learned imputers feed the identical AGCRN architecture. They are trained jointly using their native visible reconstruction/consistency mechanisms plus the common synthetic-hidden target loss and downstream objective; this joint training choice is explicit because the machinery of two-stage optimization is not specified in the manuscript. No test values after the forecast origin are supplied.

Every model uses the same prepared chronological splits, fixed validation/test mask files, shuffle seed and per-window corruption RNG. Each non-clean training window independently draws one of four fault families and one of the three nonzero rates, with 10% intentional clean windows. `training_masks.json` records realized family/rate/count for checking this policy. Validation weights clean=.1 and each of the twelve family/rate conditions=.075. Probabilistic checkpoint selection is lexicographic NLL/MAE; native point selection uses MAE then squared-error-equivalent NLL. The graph method's dropout remains .1. All use the same width 64 initial setting, Adam .001, early-stopping and gradient clipping; architecture-specific defaults such as one recurrent layer, AGCRN embedding size 10/Chebyshev support order `diffusion+1`, and native WaveNet/GCN-M four blocks by two layers are explicit adapter choices. Search budgets should be locked before formal runs.

`--objective point` retains the original point output and masked MAE training (UIGNN retains its non-quantile MSE/recovery branch); saved evaluation excludes NLL, CRPS, variance and intervals. `--objective gaussian` attaches exactly the same residual heteroscedastic output-space adapter to every deterministic forecaster: `Linear(H,2H)` on its per-node horizon vector, yielding a mean correction and positive variance. The complete network is retrained with NLL and receives its own validation-only scalar variance calibration. Only those Gaussian-adapted outputs are marked eligible for the manuscript's probabilistic table. Plot grouping keeps architecture/imputer/objective distinct and skips unavailable point-probability endpoints.

`uignn --objective evidential` additionally runs the original evidential head and NIG objective with its .5 squared-error recovery term. For the common H-step task its original one-step mean layer becomes `Linear(window,H)`, sharing the native three evidence parameters across horizons; this is an explicit multi-step adaptation. It is evaluated with the actual resulting Student-t NLL, analytic Student-t CRPS and Student-t intervals, calibrated separately on validation. It is labeled as a native-distribution diagnostic, distinct from the common Gaussian-adapter table. The CRPS formula was checked against an independent CDF-squared numerical integral.

GCN-M preserves temporal/spatial decay, temporal mean within the available input window, five-neighbor spatial statistics and a 60-window recent/day/week memory layout (`nh=1,nd=2,nw=2,tau=12` for the default 12-step input). Historical references come from a frozen source-training bank and must end before the input window; unavailable historical windows fall back to that window's actually visible values, never hidden ground truth. This frozen-memory deployment choice is saved, rather than silently reading uncorrupted test history. GCN-M uses its native forecast-only objective, while UIGNN retains its native .5 visible-recovery MSE; extra graph-method reconstruction/entropy penalties are not applied to these one-step baselines. The original dynamic graph requires at least twenty sensors. Its local-feature wrapper preserves the batch axis when invoking the original step, fixing the upstream unrestricted `squeeze` failure at batch one. The original script's unused imports refer to absent modules; only its exact required class definitions are loaded. Old Graph WaveNet/GCN-M `Conv1d` layers with two-dimensional tuple kernels are represented by shape-equivalent modern `Conv2d` modules. DCRNN's original graph utility definitions are loaded without importing its unused TensorFlow data path, and lazy parameters are instantiated before creating the optimizer. These compatibility changes retain the actual model operations.

MagiNet's exact model classes are loaded from its script without executing its top-level argument parsing/training service code. No pre-filling network substitutes for its adaptive missing encoder. Source preprocessing/graph arrays are stored in new checkpoints and mismatches are rejected at evaluation; KNN references are rebuilt from checkpointed source values on reload. Point and probabilistic methods share forecast origins, so `rgdgl-compare` can verify their paired targets.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_functional.py tests/test_native_baselines.py
```

All 17 core/native checks passed locally in 76.74 s. Functional checks use actual upstream networks on synthetic timelines, not the paper results: all thirteen combinations ran one real training epoch and evaluation through the common CLI, with finite gradients, changed parameters, exact reload and hidden-input invariance. The small native input has twenty sensors to respect GCN-M's graph bound. Batch-one GCN-M benchmarking and point-only plotting are included. Without the separately fetched sources these optional checks explicitly skip; CI fetches the pinned sources before running them. Full twelve-horizon METR-LA/PEMS-BAY, multi-seed/mask-realization, common tuning-budget and latency comparisons still require real execution.
