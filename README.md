# RGDGL

Implementation of **Reliability-Gated Dynamic Graph Learning for Missing-Sensor Traffic Imputation and Probabilistic Forecasting**.

The model combines a causal reliability-gated road-sensor graph, graph-GRU states, missing-entry reconstruction and direct Gaussian forecasts. Unavailable sources receive zero graph weight; a fully unavailable neighborhood produces a zero message and retains the local recurrent path.

## Install

Use Python 3.11 or later and install from this directory in an activated environment:

```bash
python -m pip install -e .
```

Install a compatible CUDA PyTorch build to use `--device cuda`; CPU is the default.

## Prepare traffic timelines

Obtain the METR-LA or PEMS-BAY speed data and sensor-distance tables from the DCRNN dataset distribution. The input HDF must contain a timestamp-indexed DataFrame with one sensor per column. A distance CSV has the columns `from,to,distance`; its IDs must match the HDF column names. Unlisted edges have infinite distance. Distances use one common nonnegative unit.

```bash
rgdgl prepare --input data/metr-la.h5 --distance data/distances.csv --output data/metr --realizations 10 --seed 13
```

An alternative NPZ input contains `values[time,node,channel]`, `distance[node,node]` and optional `clock[time]` as fractions of a day. A two-dimensional value array becomes one channel. When clock is absent, samples are interpreted as consecutive five-minute intervals beginning at midnight.

Finite readings are available, including numerical zeros. Declare `--sentinel VALUE` only when the dataset metadata identifies that value as unavailable. NaN/Inf and the explicit sentinel are excluded from targets before normalization. Channel statistics use available training readings only. The chronological split is 70/10/20 and windows never cross it.

Preparation saves the normalized train/validation/test timelines, source-availability masks, distance matrix and a fixed bank of global fault masks. Each family has rates 0, 0.1, 0.3 and 0.5 and ten realizations by default. All methods consume these same files.

Random masks select available entries without replacement. Blocks choose sensors, start times and durations of 6, 12 or 24 steps, truncating the final proposal to reach the target count. Persistent outages hide complete sensor timelines, so their realized rate depends on source availability. Mixed masks allocate up to a third of the target to whole sensors, a third to blocks, and the remaining count to random entries; indivisible whole-sensor leftovers enter the random allocation. No naturally absent value becomes an artificial target.

## Train a universal model

```bash
rgdgl train --data data/metr --output outputs/full --variant full --seed 13
```

Defaults are 12 input steps, 12 forecast steps, width 64, diffusion order two, Adam learning rate 0.001 and validation patience 15. Each epoch draws one global training fault family/rate and leaves 10% of windows clean. The selected model minimizes uncalibrated validation NLL over the twelve family/nonzero-rate conditions. A single multiplicative variance scale is fitted on the same validation bank and frozen for evaluation.

The candidate graph contains each sensor and its ten nearest neighbors. At recurrent step `s`, correlations and source observation fractions use only the prefix ending at `s`, capped at the input-window length. Co-observed correlations require four time/channel pairs and nonzero variance; shrinkage defaults to ten. Learned nonnegative distance and correlation coefficients define affinities. Exponentials use rowwise maximum subtraction, followed by absolute source-reliability gating. Correlation statistics are detached from gradients.

The Gaussian NLL supervises available future targets. The L1 imputation term supervises only artificially hidden, originally available input entries. The entropy term favors concentrated neighbor weights. `--imputation-weight` defaults to 1 and `--entropy-weight` to 0.001. Other controls include `--neighbors`, `--minimum-pairs`, `--shrinkage`, `--diffusion`, `--width`, `--epochs` and `--batch-size`.

## Baseline and ablation comparisons

```bash
rgdgl train --data data/metr --output outputs/static --variant static --seed 13
rgdgl train --data data/metr --output outputs/no_gate --variant no_gate --seed 13
rgdgl train --data data/metr --output outputs/no_correlation --variant no_correlation --seed 13
rgdgl train --data data/metr --output outputs/no_imputation --variant no_imputation --seed 13
```

The static-distance graph-GRU is an encoder-controlled baseline. `no_gate` removes reliability multiplication while keeping dynamic correlation, `no_correlation` removes the correlation contribution, and `no_imputation` removes reconstruction supervision. All variants retain the same feature inputs, Gaussian heads, dimensions, mask policy, optimizer and validation selection. The static control sets source reliability to one and correlation to zero; it therefore answers a different question from the dynamic graph without a reliability gate.

## Evaluate frozen models

```bash
rgdgl evaluate --data data/metr --model outputs/full/model.pt --family random --rate 0.3 --realization 0 --output outputs/full_random03
rgdgl evaluate --data data/metr --model outputs/no_gate/model.pt --family random --rate 0.3 --realization 0 --output outputs/nogate_random03
```

Evaluate the same checkpoint over all four fault families, all rates and realization IDs. `--uncalibrated` evaluates its original variance. `--mape-threshold 5` excludes targets below an absolute speed of five from MAPE; keep this threshold and speed units fixed across comparisons.

`metrics.json` contains MAE/RMSE at 15, 30 and 60 minutes and over the full horizon; thresholded MAPE; hidden-entry imputation MAE; Gaussian NLL including its normalizing constant in original speed units; closed-form Gaussian CRPS; empirical coverage, coverage error and width at nominal levels 0.50–0.95. Clean-input imputation MAE is null because no artificial imputation targets exist. Parameter count and mean per-window inference time are reported separately; the latter is measured under the requested batch size and should be interpreted as throughput-derived latency.

`predictions.npz` stores forecasts, variances, targets and availability at each origin. Overlapping forecast windows are retained, not treated as independent experimental runs. `model.pt` stores the selected parameters, exact training options and validation variance scale. `history.json` records actual train and development losses.

```bash
rgdgl plot --metrics outputs/full_random03/metrics.json outputs/nogate_random03/metrics.json --output outputs/figures
```

Supply multiple rates from one dataset, fault family and seed to draw robustness curves. `robustness_slopes.json` fits MAE against realized missingness for each variant. Keep conditions separate when making these plots; repeat training seeds and complete mask realizations before summarizing variability.
