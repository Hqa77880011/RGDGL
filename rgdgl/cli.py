import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr, ndtri, betaln
from scipy.stats import t as student_t
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from .data import TrafficWindows, fault_mask, prepare
from .model import ReliabilityGraph, loss


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def student_parameters(values,calibration=1.,scale=1.):
    raw=values["nig"]
    alpha=raw[:,None,0,:,None]+1
    beta=raw[:,None,1,:,None]+.3
    precision=raw[:,None,2,:,None]+1
    df=2*alpha
    radius=np.sqrt(beta*(1+precision)/(alpha*precision)*calibration)*scale
    return df,radius


def validation_nll(values,calibration=1.):
    selected=values["available"].astype(bool)
    if "nig" in values:
        df,radius=student_parameters(values,calibration)
        value=-student_t.logpdf((values["target"]-values["mean"])/radius,df)+np.log(radius)
    else:
        variance=values["variance"]*calibration
        value=.5*(np.log(2*np.pi*variance)+(values["target"]-values["mean"])**2/variance)
    return float(value[selected].mean())


def construct(args, metadata):
    if getattr(args,"forecaster","full")!="full":
        from .baselines import NativeForecast
        return NativeForecast(args,metadata,read(Path(args.data)/"train.npz")).to(args.device)
    return ReliabilityGraph(metadata["distance"], metadata["mean"].size, args.width, args.horizon,
                            args.neighbors, args.window, args.minimum_pairs, args.shrinkage,
                            args.diffusion, args.variant, getattr(args, "dropout", 0.1)).to(args.device)


def move(value,device):
    return {key:item.to(device) for key,item in value.items()} if isinstance(value,dict) else value.to(device)


def windows(data,hidden,args,require_input=False):
    return TrafficWindows(data,hidden,args.window,args.horizon,require_input=require_input,context=getattr(args,"forecaster",None)=="gcnm")


@torch.no_grad()
def collect(model, dataset, args):
    model.eval()
    arrays = {key: [] for key in ("mean", "variance", "target", "available", "imputation", "past", "hidden")}
    elapsed, count = 0.0, 0
    for batch in DataLoader(dataset, batch_size=args.batch_size):
        x, b, q, clock, y, available = [move(value,args.device) for value in batch]
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
        if getattr(model,"objective_name",None)=="evidential":arrays.setdefault("nig",[]).append(model.last_evidence[:,-3:,:].cpu().numpy())
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
    history, best, waiting = [], (np.inf, np.inf), 0
    validation_bank = [windows(validation,np.zeros_like(validation["available"]),args)]
    validation_weights=[.1]+[.9/12]*12
    for family in ("random", "block", "persistent", "mixed"):
        for rate in (0.1, 0.3, 0.5):
            hidden = read(root / f"val_{family}_{rate:.1f}_0.npz")["hidden"]
            validation_bank.append(windows(validation, hidden, args))
    corruption_audit=[]
    for epoch in range(args.epochs):
        dataset = windows(training,np.zeros_like(training["available"]),args,require_input=True)
        if not len(dataset):
            raise ValueError("Training needs windows with observed inputs and available forecast targets")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        losses = []
        for batch in loader:
            values = [move(item,args.device) for item in batch]
            clean = torch.tensor(rng.random(len(values[0])) < 0.1, device=args.device)
            values[2][clean] = 0
            corruption_audit.extend(dict(epoch=epoch+1,family="clean",rate=0.,hidden=0) for _ in range(int(clean.sum())))
            # Preserve intentional clean windows. Other empty masks are resampled
            # conditionally on at least one eligible reconstruction target.
            for index in (~clean).nonzero().flatten().tolist():
                available = values[1][index].cpu().numpy().astype(bool)
                family=rng.choice(["random","block","persistent","mixed"])
                rate=float(rng.choice([.1,.3,.5]))
                sampled=fault_mask(available,family,rate,int(rng.integers(2**31)),nonempty=True)
                values[2][index]=torch.as_tensor(sampled,device=args.device)
                corruption_audit.append(dict(epoch=epoch+1,family=str(family),rate=rate,hidden=int(sampled.sum())))
            objective = loss(model, *values, args.imputation_weight, args.entropy_weight)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training objective")
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(objective.detach()))
        residuals, variances, errors,records = [], [], [],[]
        for dataset in validation_bank:
            values, _ = collect(model, dataset, args)
            records.append(values)
            valid = values["available"].astype(bool)
            residuals.append((values["mean"] - values["target"])[valid] ** 2)
            variances.append(values["variance"][valid])
            errors.append(np.abs(values["mean"] - values["target"])[valid])
        nll=float(sum(weight*validation_nll(record) for weight,record in zip(validation_weights,records)))
        mae=float(sum(weight*error.mean() for weight,error in zip(validation_weights,errors)))
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_nll=nll, validation_mae=mae))
        score=(mae,nll) if getattr(args,"objective",None)=="point" else (nll,mae)
        if score < best:
            best, waiting = score, 0
            calibration=max(float(sum(weight*np.mean(residual/var) for weight,residual,var in zip(validation_weights,residuals,variances))),1e-6)
            if getattr(args,"objective",None)=="evidential":
                fitted=minimize_scalar(lambda log_c:sum(weight*validation_nll(record,float(np.exp(log_c))) for weight,record in zip(validation_weights,records)),bounds=(-8,8),method="bounded")
                if not fitted.success:raise RuntimeError("Native Student-t calibration failed")
                calibration=float(np.exp(fitted.x))
            torch.save(dict(state=model.state_dict(), config=vars(args), variance_scale=calibration,preprocessing={key:torch.as_tensor(value) for key,value in metadata.items()}), output / "model.pt")
        else:
            waiting += 1
            if waiting >= args.patience:
                break
    (output/"training_masks.json").write_text(json.dumps(corruption_audit,indent=2),encoding="utf-8")
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
    distribution="Gaussian"
    nll=.5*(np.log(2*np.pi*variance)+error**2/variance)
    if "nig" in values:
        distribution="native NIG Student-t"
        df,radius=student_parameters(values,calibration,scale)
        z=(truth-prediction)/radius
        nll=-student_t.logpdf(z,df)+np.log(radius)
        constant=2*np.sqrt(df)/(df-1)*np.exp(betaln(.5,df-.5)-2*betaln(.5,df/2))
        crps=radius*(z*(2*student_t.cdf(z,df)-1)+2*student_t.pdf(z,df)*(df+z*z)/(df-1)-constant)
    result = dict(distribution=distribution,mae=float(np.abs(error)[available].mean()), rmse=float(np.sqrt((error ** 2)[available].mean())),
                  nll=float(nll[available].mean()),
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
        interval_radius=student_t.ppf((1+nominal)/2,df)*radius if "nig" in values else ndtri((1+nominal)/2)*std
        coverage = float((np.abs(error) <= interval_radius)[available].mean())
        result["intervals"].append(dict(nominal=float(nominal), coverage=coverage,
                                        coverage_error=abs(coverage - nominal), width=float(np.broadcast_to(2*interval_radius,error.shape)[available].mean())))
    result["mean_coverage_error"] = float(np.mean([row["coverage_error"] for row in result["intervals"]]))
    return result, dict(mean=prediction, variance=variance, target=truth, available=available,**({"nig":values["nig"]} if "nig" in values else {}))


def evaluate(args):
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    config = argparse.Namespace(**checkpoint["config"])
    config.device, config.batch_size,config.data = args.device, args.batch_size,args.data
    metadata, test = read(root / "metadata.npz"), read(root / "test.npz")
    if "preprocessing" in checkpoint and any(not np.array_equal(metadata[key],value.cpu().numpy(),equal_nan=True) for key,value in checkpoint["preprocessing"].items()):
        raise ValueError("Prepared normalization/graph differ from the frozen checkpoint")
    model = construct(config, metadata)
    model.load_state_dict(checkpoint["state"])
    mask_record = read(root / f"test_{args.family}_{args.rate:.1f}_{args.realization}.npz")
    dataset = windows(test, mask_record["hidden"], config)
    values, latency = collect(model, dataset, config)
    calibration = checkpoint["variance_scale"] if not args.uncalibrated else 1.0
    result, predictions = metrics(values, metadata["mean"], metadata["scale"], calibration, args.mape_threshold)
    predictions["origin_indices"] = np.asarray(dataset.origins)
    objective=getattr(config,"objective","gaussian")
    if objective=="point":
        for key in ("nll","crps","intervals","mean_coverage_error","distribution"):result.pop(key,None)
        predictions.pop("variance",None)
    result.update(forecaster=getattr(config,"forecaster","full"),imputer=getattr(config,"imputer",None),objective=objective,
                  probabilistic_table_eligible=objective=="gaussian",config=vars(args), model_variant=config.variant if getattr(config,"forecaster","full")=="full" else f"{config.imputer}+{config.forecaster}:{objective}", parameters=sum(p.numel() for p in model.parameters()),
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
    for name, default in (("lr", 0.001), ("dropout", 0.1), ("imputation-weight", 1.0), ("entropy-weight", 0.001), ("shrinkage", 10.0)):
        training.add_argument(f"--{name}", type=float, default=default)
    training.add_argument("--device", default="cpu")
    training.add_argument("--forecaster",choices=["full","dcrnn","graphwavenet","agcrn","agcrn_mask","gcnm","uignn"],default="full")
    training.add_argument("--imputer",choices=["mean","forward","knn","brits","saits","grin","maginet"],default="mean")
    training.add_argument("--objective",choices=["point","gaussian","evidential"],default="gaussian")
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
        if args.forecaster=="full" and args.objective!="gaussian":parser.error("The main probabilistic method uses Gaussian NLL")
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
            eligible=[row for row in records if metric in row and (metric in {"mae","rmse"} or row.get("probabilistic_table_eligible",True))]
            if not eligible:continue
            figure, axis = plt.subplots(figsize=(7, 4))
            for variant in sorted(set(row["model_variant"] for row in eligible)):
                group = sorted([row for row in eligible if row["model_variant"] == variant], key=lambda row: row["realized_missing_rate"])
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
