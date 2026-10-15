import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from .cli import construct, read
from .data import TrafficWindows


def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("Positive repeats and nonnegative warmup required")
    root = Path(args.data)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    options = argparse.Namespace(**checkpoint["config"])
    options.device = args.device
    model = construct(options, read(root / "metadata.npz"))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    data = read(root / "test.npz")
    hidden = read(root / "test_random_0.3_0.npz")["hidden"]
    dataset = TrafficWindows(data, hidden, options.window, options.horizon)
    if not len(dataset):
        raise ValueError("Benchmark requires one available forecast window")
    x, available, q, clock, _, _ = [torch.as_tensor(value, device=args.device)[None] for value in dataset[0]]
    inputs = x, available * (1 - q), clock
    process = psutil.Process()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(args.warmup):
            model(*inputs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        measured, peak_rss = [], process.memory_info().rss
        for _ in range(args.repeats):
            start = time.perf_counter()
            model(*inputs)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            measured.append(time.perf_counter() - start)
            peak_rss = max(peak_rss, process.memory_info().rss)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            model(*inputs)
    result = dict(parameters=sum(parameter.numel() for parameter in model.parameters()),
                  batch_one_latency_ms=float(np.mean(measured) * 1000),
                  batch_one_latency_p95_ms=float(np.quantile(measured, 0.95) * 1000),
                  sampled_process_peak_rss_mib=peak_rss / 1024 ** 2,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2 if args.device.startswith("cuda") else None,
                  profiler_supported_operator_flops=sum(event.flops for event in profile.key_averages()), config=vars(args))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")


def compare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", nargs="+", required=True)
    parser.add_argument("--second", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block", type=int, default=288)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if len(args.first) != len(args.second) or min(args.block, args.resamples) < 1:
        parser.error("Paired complete runs and positive bootstrap sizes required")
    differences = []
    for first, second in zip(args.first, args.second):
        a, b = read(first), read(second)
        if not np.array_equal(a["origin_indices"], b["origin_indices"]) or not np.array_equal(a["target"], b["target"]) or not np.array_equal(a["available"], b["available"]):
            raise ValueError("Seed/mask runs must have identical forecast origins and targets")
        observed = a["available"].astype(bool)
        error = (np.abs(a["mean"] - a["target"]) - np.abs(b["mean"] - b["target"])) * observed
        axes = tuple(range(1, error.ndim))
        differences.append((error.sum(axes), observed.sum(axes)))
    rng = np.random.default_rng(args.seed)
    estimates = []
    for _ in range(args.resamples):
        selected_runs = rng.integers(len(differences), size=len(differences))
        run_means = []
        for run in selected_runs:
            numerator, denominator = differences[int(run)]
            block = min(args.block, len(numerator))
            starts = rng.integers(len(numerator) - block + 1, size=int(np.ceil(len(numerator) / block)))
            positions = np.concatenate([np.arange(start, start + block) for start in starts])[:len(numerator)]
            run_means.append(float(numerator[positions].sum() / max(1, denominator[positions].sum())))
        estimates.append(float(np.mean(run_means)))
    measured = float(np.mean([numerator.sum() / denominator.sum() for numerator, denominator in differences]))
    result = dict(mean_mae_difference=measured, confidence_interval_95=np.quantile(estimates, [0.025, 0.975]).tolist(),
                  complete_runs=len(differences), block_origins=args.block, resamples=args.resamples,
                  centered_bootstrap_two_sided_p=float((np.sum(np.abs(np.asarray(estimates) - measured) >= abs(measured)) + 1) / (args.resamples + 1)))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
