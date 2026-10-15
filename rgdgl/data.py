from pathlib import Path

import numpy as np
import pandas as pd


def fault_mask(available, family, rate, seed):
    rng = np.random.default_rng(seed)
    hidden = np.zeros_like(available, dtype=bool)
    target = round(rate * int(available.sum()))
    if family == "persistent":
        nodes = rng.choice(available.shape[1], round(rate * available.shape[1]), replace=False)
        hidden[:, nodes] = available[:, nodes]
        return hidden
    if family == "mixed":
        budget = target // 3
        for node in rng.permutation(available.shape[1]):
            proposal = available[:, node]
            if int(proposal.sum()) <= budget:
                hidden[:, node] = proposal
                budget -= int(proposal.sum())
        block_target = int(hidden.sum()) + target // 3
    else:
        block_target = target if family == "block" else 0
    attempts = 0
    while hidden.sum() < block_target:
        node = int(rng.integers(available.shape[1]))
        start = int(rng.integers(len(available)))
        duration = int(rng.choice([6, 12, 24]))
        proposal = np.zeros_like(hidden)
        proposal[start:start + duration, node] = True
        positions = np.flatnonzero(proposal & available & ~hidden)
        remaining = block_target - int(hidden.sum())
        hidden.flat[positions[:remaining]] = True
        attempts += 1
        if attempts > 100 * max(1, len(available) * available.shape[1]):
            raise RuntimeError("Insufficient eligible entries for block mask")
    remaining = target - int(hidden.sum())
    if remaining:
        positions = np.flatnonzero(available & ~hidden)
        hidden.flat[rng.choice(positions, remaining, replace=False)] = True
    return hidden


def prepare(args):
    source = Path(args.input)
    if source.suffix in (".h5", ".hdf"):
        frame = pd.read_hdf(source)
        values = frame.to_numpy(dtype=np.float32)[:, :, None]
        ids = frame.columns.astype(str).to_numpy()
        clock = (frame.index.hour * 60 + frame.index.minute).to_numpy() / 1440
        if not args.distance:
            raise ValueError("HDF input needs a sensor distance CSV")
        distance_table = pd.read_csv(args.distance, dtype={"from": str, "to": str})
        lookup = {identity: i for i, identity in enumerate(ids)}
        distance = np.full((len(ids), len(ids)), np.inf, dtype=np.float32)
        np.fill_diagonal(distance, 0)
        for row in distance_table.itertuples(index=False):
            if row[0] in lookup and row[1] in lookup:
                distance[lookup[row[0]], lookup[row[1]]] = float(row[2])
    else:
        with np.load(source, allow_pickle=False) as data:
            values, distance = data["values"].astype(np.float32), data["distance"].astype(np.float32)
            clock = data["clock"] if "clock" in data else np.arange(len(values)) % 288 / 288
        if values.ndim == 2:
            values = values[..., None]
    if values.ndim != 3 or distance.shape != (values.shape[1], values.shape[1]):
        raise ValueError("Expected values[time,node,channel] and aligned distance[node,node]")
    available = np.isfinite(values)
    if args.sentinel is not None:
        available &= values != args.sentinel
    boundaries = [0, int(0.7 * len(values)), int(0.8 * len(values)), len(values)]
    observed = np.where(available[:boundaries[1]], values[:boundaries[1]], np.nan)
    mean = np.nanmean(observed, axis=(0, 1), keepdims=True)
    scale = np.nanstd(observed, axis=(0, 1), keepdims=True).clip(1e-6)
    if not np.isfinite(mean).all():
        raise ValueError("Every channel needs finite training observations")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "metadata.npz", mean=mean, scale=scale, distance=distance)
    for part, left, right in zip(("train", "val", "test"), boundaries[:-1], boundaries[1:]):
        b = available[left:right]
        x = np.where(b, (values[left:right] - mean) / scale, 0).astype(np.float32)
        np.savez_compressed(output / f"{part}.npz", x=x, available=b, clock=clock[left:right])
        for family_number, family in enumerate(("random", "block", "persistent", "mixed")):
            for rate_number, rate in enumerate((0.0, 0.1, 0.3, 0.5)):
                for realization in range(args.realizations):
                    seed = args.seed + left + 100000 * family_number + 1000 * rate_number + realization
                    q = fault_mask(b, family, rate, seed)
                    np.savez_compressed(output / f"{part}_{family}_{rate:.1f}_{realization}.npz", hidden=q, seed=seed,
                                        realized_rate=q.sum() / max(1, b.sum()))


class TrafficWindows:
    def __init__(self, data, hidden, window=12, horizon=12):
        self.data, self.hidden, self.window, self.horizon = data, hidden, window, horizon
        self.origins = [i for i in range(window, len(data["x"]) - horizon + 1) if data["available"][i:i + horizon].any()]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        end = self.origins[index]
        start = end - self.window
        clock = self.data["clock"][start:end]
        time = np.stack([np.sin(2 * np.pi * clock), np.cos(2 * np.pi * clock)], 1).astype(np.float32)
        return (self.data["x"][start:end], self.data["available"][start:end].astype(np.float32),
                self.hidden[start:end].astype(np.float32), time, self.data["x"][end:end + self.horizon],
                self.data["available"][end:end + self.horizon].astype(np.float32))
