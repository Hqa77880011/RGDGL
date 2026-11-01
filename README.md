# RGDGL

**Reliability-Gated Dynamic Graph Learning for Missing-Sensor Traffic Imputation and Probabilistic Forecasting**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

## Data and preprocessing

Obtain METR-LA or PEMS-BAY speed data and sensor distances from the [DCRNN data distribution](https://github.com/liyaguang/DCRNN#data-preparation). The HDF contains a timestamp-indexed DataFrame with one sensor per column. Distance CSV columns are `from,to,distance`, using matching sensor IDs and one nonnegative distance unit. Unlisted edges have infinite distance.

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

| Component | Entry point | Command/configuration |
| --- | --- | --- |
| Causal correlation, reliability gate, graph GRU, dual heads | `rgdgl/model.py:ReliabilityGraph` | `rgdgl train --variant full` |
| Masked reconstruction + Gaussian NLL + entropy | `rgdgl/model.py:loss` | `--imputation-weight 1 --entropy-weight .001` |
| Chronological train-only scaling and global fault bank | `rgdgl/data.py:prepare` | `rgdgl prepare` |
| MAE/RMSE/MAPE, reconstruction MAE, NLL/CRPS and coverage/width | `rgdgl/cli.py:metrics` | `rgdgl evaluate` |
| Robustness slopes, paired block bootstrap, batch-one costs | `rgdgl/cli.py:main`, `rgdgl/analysis.py` | `rgdgl plot`, `rgdgl-compare`, `rgdgl-benchmark` |

The graph model uses dropout 0.1 during training and lexicographic validation (NLL, MAE). Training draws 10% clean windows; other windows require a nonempty eligible reconstruction target. Empty non-clean masks are resampled within the same fault family; completely unobserved input windows are excluded. A rounded zero missingness budget hides one eligible reading or persistent-fault sensor. Validation/test mask banks stay fixed.

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

UIGNN's optional [sample-data archive](https://github.com/lijunxian111/UIGNN/blob/56485b647886f4fd12ccc4fce5d66f5e49f526ea/data/sedata/mat.zip) is separate from the shared traffic inputs. After fetching the source, restore that archive when needed:

```bash
git -C .external/UIGNN restore --source=56485b647886f4fd12ccc4fce5d66f5e49f526ea -- data/sedata/mat.zip
```

`baseline_sources.json` lists source revisions and licenses. `python scripts/fetch_baselines.py` obtains pinned sparse checkouts under `.external/`. UIGNN and GRIN are separate runtime dependencies with no top-level reuse license in those revisions; their source is not redistributed.

| Forecaster/imputer | Source and retained mechanism |
| --- | --- |
| `dcrnn` | [DCRNN PyTorch port](https://github.com/chnsh/DCRNN_PyTorch): bidirectional diffusion-GRU encoder/decoder |
| `graphwavenet` | [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet): dilated gated temporal convolutions, skip/residual paths and learned adaptive adjacency |
| `agcrn`, `agcrn_mask` | [AGCRN](https://github.com/LeiBAI/AGCRN): node-adaptive parameter pools/graph recurrent cells; the mask control adds the actual observation channel |
| `brits` + AGCRN | BRITS recurrent temporal/feature regression and forward/backward consistency from the [GRIN reference implementation](https://github.com/Graph-Machine-Learning-Group/grin) |
| `saits` + AGCRN | [SAITS](https://github.com/WenjieDu/SAITS): both diagonal-masked self-attention blocks, attention-weighted combination, observed reconstruction and hidden-target losses |
| `grin` + AGCRN | [GRIN](https://github.com/Graph-Machine-Learning-Group/grin): bidirectional graph recurrent imputation and spatial decoding |
| `maginet` + AGCRN | [MagiNet](https://github.com/JeremyChou28/MagiNet): original adaptive missing encoder and complete mask-aware spatio-temporal decoder |
| `gcnm` | [GCN-M](https://github.com/JingweiZuo/GCN-M): original eight-feature local module, attention historical memory and dynamic graph forecaster |
| `uignn` | [UIGNN](https://github.com/lijunxian111/UIGNN): original inductive diffusion network, recovery objective and optional NIG evidential head/loss |

`mean`, `forward` and five-neighbor `knn` pair with DCRNN, Graph WaveNet or AGCRN. Training data fit means/KNN reference records; filling and KNN use only the input prefix. Learned imputers feed the same AGCRN and train jointly with native reconstruction/consistency, synthetic-hidden-target loss and the downstream objective.

Every model uses the same prepared chronological splits, fixed validation/test mask files, shuffle seed and per-window corruption RNG. Each non-clean training window independently draws one of four fault families and one of the three nonzero rates, with 10% intentional clean windows. `training_masks.json` records realized family/rate/count for checking this policy. Validation weights clean=.1 and each of the twelve family/rate conditions=.075. Probabilistic checkpoint selection is lexicographic NLL/MAE; native point selection uses MAE then squared-error-equivalent NLL. The graph method's dropout remains .1. All use the same width 64 initial setting, Adam .001, early-stopping and gradient clipping; architecture-specific defaults such as one recurrent layer, AGCRN embedding size 10/Chebyshev support order `diffusion+1`, and native WaveNet/GCN-M four blocks by two layers are explicit adapter choices. Search budgets should be locked before formal runs.

`--objective point` retains the original point output and masked MAE training (UIGNN retains its non-quantile MSE/recovery branch); saved evaluation excludes NLL, CRPS, variance and intervals. `--objective gaussian` attaches exactly the same residual heteroscedastic output-space adapter to every deterministic forecaster: `Linear(H,2H)` on its per-node horizon vector, yielding a mean correction and positive variance. The complete network is retrained with NLL and receives its own validation-only scalar variance calibration. Gaussian-adapted outputs support the common probabilistic comparisons. Plot grouping keeps architecture/imputer/objective distinct and skips unavailable point-probability endpoints.

`uignn --objective evidential` additionally runs the original evidential head and NIG objective with its .5 squared-error recovery term. For the common H-step task its original one-step mean layer becomes `Linear(window,H)`, sharing the native three evidence parameters across horizons; this is an explicit multi-step adaptation. It is evaluated with the actual resulting Student-t NLL, analytic Student-t CRPS and Student-t intervals, calibrated separately on validation. It is labeled as a native-distribution diagnostic, distinct from the common Gaussian-adapter table. The CRPS formula was checked against an independent CDF-squared numerical integral.

GCN-M uses temporal/spatial decay, available-window temporal means, five-neighbor spatial statistics and a 60-window recent/day/week memory (`nh=1,nd=2,nw=2,tau=12` at the default window). References come from a frozen training bank and end before the input window; missing references use its visible values. It requires at least twenty sensors and uses its forecast-only objective. UIGNN retains its 0.5 visible-recovery MSE. GCN-M preserves the batch axis, legacy tuple-kernel convolutions use shape-equivalent Conv2d, and DCRNN lazy parameters initialize before optimizer creation.

MagiNet loads model definitions without the source script's top-level training code. Its Chebyshev graph basis is checkpointed as buffers. Checkpoints also store preprocessing/graph arrays; evaluation verifies them and rebuilds KNN references from saved training values. `rgdgl-compare` checks paired targets and forecast origins.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_functional.py tests/test_native_baselines.py
```
