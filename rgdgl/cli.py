import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from torch.utils.data import DataLoader

from .data import TrafficWindows, fault_mask, prepare
from .model import ReliabilityGraph, loss


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def construct(args, metadata):
    return ReliabilityGraph(metadata["distance"], metadata["mean"].size, args.width, args.horizon,
                            args.neighbors, args.window, args.minimum_pairs, args.shrinkage,
                            args.diffusion, args.variant).to(args.device)


@torch.no_grad()
def collect(model, dataset, args):
    model.eval()
    arrays = {key: [] for key in ("mean", "variance", "target", "available", "imputation", "past", "hidden")}
    elapsed, count = 0.0, 0
    for batch in DataLoader(dataset, batch_size=args.batch_size):
        x, b, q, clock, y, available = [value.to(args.device) for value in batch]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        reconstruction, mean, variance, _ = model(x, b * (1 - q), clock)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        count += len(x)
        for key, value in zip(arrays, (mean, variance, y, available, reconstruction, x, b * q)):
            arrays[key].append(value.cpu().numpy())
    if count == 0:
        raise ValueError("No forecast windows in this split")
    return {key: np.concatenate(value) for key, value in arrays.items()}, elapsed / count


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = read(root / "metadata.npz")
    training, validation = read(root / "train.npz"), read(root / "val.npz")
    model = construct(args, metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best, waiting = [], np.inf, 0
    validation_bank = []
    for family in ("random", "block", "persistent", "mixed"):
        for rate in (0.1, 0.3, 0.5):
            hidden = read(root / f"val_{family}_{rate:.1f}_0.npz")["hidden"]
            validation_bank.append(TrafficWindows(validation, hidden, args.window, args.horizon))
    for epoch in range(args.epochs):
        family = rng.choice(["random", "block", "persistent", "mixed"])
        rate = float(rng.choice([0.1, 0.3, 0.5]))
        hidden = fault_mask(training["available"], family, rate, args.seed + epoch)
        dataset = TrafficWindows(training, hidden, args.window, args.horizon)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        losses = []
        for batch in loader:
            values = [item.to(args.device) for item in batch]
            clean = torch.tensor(rng.random(len(values[0])) < 0.1, device=args.device)
            values[2][clean] = 0
            objective = loss(model, *values, args.imputation_weight, args.entropy_weight)
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(objective.detach()))
        residuals, variances, errors = [], [], []
        for dataset in validation_bank:
            values, _ = collect(model, dataset, args)
            valid = values["available"].astype(bool)
            residuals.append((values["mean"] - values["target"])[valid] ** 2)
            variances.append(values["variance"][valid])
            errors.append(np.abs(values["mean"] - values["target"])[valid])
        squared, variance = np.concatenate(residuals), np.concatenate(variances)
        nll = float(np.mean(0.5 * (np.log(2 * np.pi * variance) + squared / variance)))
        mae = float(np.concatenate(errors).mean())
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_nll=nll, validation_mae=mae))
        if nll < best:
            best, waiting = nll, 0
            calibration = max(float(np.mean(squared / variance)), 1e-6)
            torch.save(dict(state=model.state_dict(), config=vars(args), variance_scale=calibration), output / "model.pt")
        else:
            waiting += 1
            if waiting >= args.patience:
                break
    (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")


def metrics(values, mean, scale, calibration, mape_threshold):
    truth = values["target"] * scale + mean
    prediction = values["mean"] * scale + mean
    variance = values["variance"] * scale ** 2 * calibration
    available = values["available"].astype(bool)
    error = prediction - truth
    std = np.sqrt(variance)
    standardized = (truth - prediction) / std
    crps = std * (standardized * (2 * ndtr(standardized) - 1) + 2 * np.exp(-standardized ** 2 / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    result = dict(mae=float(np.abs(error)[available].mean()), rmse=float(np.sqrt((error ** 2)[available].mean())),
                  nll=float((0.5 * (np.log(2 * np.pi * variance) + error ** 2 / variance))[available].mean()),
                  crps=float(crps[available].mean()))
    eligible = available & (np.abs(truth) >= mape_threshold)
    result["mape_percent"] = float(100 * np.mean(np.abs(error[eligible] / truth[eligible]))) if eligible.any() else None
    mask = values["hidden"].astype(bool)
    reconstruction_error = (values["imputation"] - values["past"]) * scale
    result["imputation_mae"] = float(np.abs(reconstruction_error)[mask].mean()) if mask.any() else None
    result["horizons"] = {}
    for index in (2, 5, 11):
        if index < error.shape[1]:
            valid = available[:, index]
            if valid.any():
                result["horizons"][str((index + 1) * 5)] = dict(mae=float(np.abs(error[:, index])[valid].mean()),
                                                            rmse=float(np.sqrt((error[:, index] ** 2)[valid].mean())))
    result["intervals"] = []
    for nominal in np.arange(0.5, 0.951, 0.05):
        radius = ndtri((1 + nominal) / 2) * std
        coverage = float((np.abs(error) <= radius)[available].mean())
        result["intervals"].append(dict(nominal=float(nominal), coverage=coverage,
                                        coverage_error=abs(coverage - nominal), width=float((2 * radius)[available].mean())))
    result["mean_coverage_error"] = float(np.mean([row["coverage_error"] for row in result["intervals"]]))
    return result, dict(mean=prediction, variance=variance, target=truth, available=available)


def evaluate(args):
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    config = argparse.Namespace(**checkpoint["config"])
    config.device, config.batch_size = args.device, args.batch_size
    metadata, test = read(root / "metadata.npz"), read(root / "test.npz")
    model = construct(config, metadata)
    model.load_state_dict(checkpoint["state"])
    mask_record = read(root / f"test_{args.family}_{args.rate:.1f}_{args.realization}.npz")
    dataset = TrafficWindows(test, mask_record["hidden"], config.window, config.horizon)
    values, latency = collect(model, dataset, config)
    calibration = checkpoint["variance_scale"] if not args.uncalibrated else 1.0
    result, predictions = metrics(values, metadata["mean"], metadata["scale"], calibration, args.mape_threshold)
    result.update(config=vars(args), model_variant=config.variant, parameters=sum(p.numel() for p in model.parameters()),
                  mean_sample_latency_seconds=latency, realized_missing_rate=float(mask_record["realized_rate"]),
                  variance_scale=calibration, seed=config.seed)
    np.savez_compressed(output / "predictions.npz", **predictions)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--input", required=True)
    data.add_argument("--distance")
    data.add_argument("--output", required=True)
    data.add_argument("--sentinel", type=float)
    data.add_argument("--realizations", type=int, default=10)
    data.add_argument("--seed", type=int, default=13)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--variant", choices=["full", "no_gate", "no_correlation", "static", "no_imputation"], default="full")
    for name, default in (("window", 12), ("horizon", 12), ("width", 64), ("neighbors", 10), ("minimum-pairs", 4),
                          ("diffusion", 2), ("epochs", 100), ("patience", 15), ("batch-size", 64), ("seed", 13)):
        training.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("lr", 0.001), ("imputation-weight", 1.0), ("entropy-weight", 0.001), ("shrinkage", 10.0)):
        training.add_argument(f"--{name}", type=float, default=default)
    training.add_argument("--device", default="cpu")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--family", choices=["random", "block", "persistent", "mixed"], default="random")
    evaluation.add_argument("--rate", type=float, choices=[0, 0.1, 0.3, 0.5], default=0.3)
    evaluation.add_argument("--realization", type=int, default=0)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--batch-size", type=int, default=64)
    evaluation.add_argument("--mape-threshold", type=float, default=5)
    evaluation.add_argument("--uncalibrated", action="store_true")
    plotting = sub.add_parser("plot")
    plotting.add_argument("--metrics", nargs="+", required=True)
    plotting.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        if min(args.window, args.horizon, args.batch_size, args.epochs, args.patience, args.width) < 1:
            parser.error("Positive window, horizon and training sizes required")
        train(args)
    elif args.command == "evaluate":
        if args.mape_threshold <= 0:
            parser.error("MAPE threshold must be positive")
        evaluate(args)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics]
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        slopes = []
        for metric in ("mae", "rmse", "nll", "crps", "mean_coverage_error"):
            figure, axis = plt.subplots(figsize=(7, 4))
            for variant in sorted(set(row["model_variant"] for row in records)):
                group = sorted([row for row in records if row["model_variant"] == variant], key=lambda row: row["realized_missing_rate"])
                x = [row["realized_missing_rate"] for row in group]
                axis.plot(x, [row[metric] for row in group], "o-", label=variant)
                if metric == "mae" and len(set(x)) > 1:
                    slopes.append(dict(variant=variant, mae_slope=float(np.polyfit(x, [row[metric] for row in group], 1)[0])))
            axis.set(xlabel="Realized missing fraction", ylabel=metric)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f"{metric}.png", dpi=180)
            plt.close(figure)
        (output / "robustness_slopes.json").write_text(json.dumps(slopes, indent=2), encoding="utf-8")
