"""CPU functional checks with synthetic traffic timelines."""
import sys

import numpy as np
import torch

from rgdgl.cli import main, metrics
from rgdgl.model import ReliabilityGraph, loss


def test_cpu_pipeline(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    rng = np.random.default_rng(13)
    values = rng.normal(30, 3, (160, 3, 1)).astype("float32")
    values[0, 0, 0] = np.nan
    distance = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :]).astype("float32")
    source = tmp_path / "source.npz"
    np.savez(source, values=values, distance=distance)
    def command(*args):
        monkeypatch.setattr(sys, "argv", ["rgdgl", *map(str, args)])
        main()
    data, run, evaluation = [tmp_path / name for name in ("data", "run", "evaluation")]
    command("prepare", "--input", source, "--output", data, "--realizations", 1)
    with np.load(data / "metadata.npz") as meta:
        np.testing.assert_allclose(meta["mean"], np.nanmean(values[:112], axis=(0, 1), keepdims=True), rtol=1e-6)
    command("train", "--data", data, "--output", run, "--window", 4, "--horizon", 2,
            "--width", 4, "--neighbors", 2, "--epochs", 1, "--batch-size", 16)
    command("evaluate", "--data", data, "--model", run / "model.pt", "--output", evaluation)
    assert (evaluation / "predictions.npz").is_file()
    command("plot", "--metrics", evaluation / "metrics.json", "--output", tmp_path / "figures")
    for variant in ("full", "no_gate", "no_correlation", "static", "no_imputation"):
        model = ReliabilityGraph(distance, width=4, horizon=2, neighbors=2, minimum_pairs=2, variant=variant)
        x = torch.randn(2, 4, 3, 1)
        available = torch.ones_like(x)
        hidden = torch.zeros_like(x)
        hidden[:, 2, 1] = 1
        clock = torch.randn(2, 4, 2)
        y = torch.randn(2, 2, 3, 1)
        before = model.mean.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        objective = loss(model, x, available, hidden, clock, y, torch.ones_like(y))
        assert torch.isfinite(objective)
        objective.backward()
        assert model.mean.weight.grad.abs().sum() > 0
        if variant == "full":
            assert model.alpha.grad.abs() > 0
            assert model.beta.grad.abs() > 0
        optimizer.step()
        assert not torch.equal(before, model.mean.weight)
        model.eval()
        expected = model(x, available, clock)[1]
        path = tmp_path / (variant + ".pt")
        torch.save(model.state_dict(), path)
        restored = ReliabilityGraph(distance, width=4, horizon=2, neighbors=2, minimum_pairs=2, variant=variant)
        restored.load_state_dict(torch.load(path, weights_only=True))
        restored.eval()
        torch.testing.assert_close(restored(x, available, clock)[1], expected, rtol=0, atol=0)
        assert all(torch.isfinite(v).all() for v in restored(x, torch.zeros_like(x), clock))


def test_metric_arithmetic():
    target = np.zeros((1, 2, 1, 1))
    values = dict(target=target, mean=target + 2, variance=target + 1,
                  available=np.ones_like(target), hidden=np.ones_like(target),
                  imputation=target + 3, past=target)
    result, _ = metrics(values, 0, 1, 1, 5)
    assert result["mae"] == result["rmse"] == 2
    assert result["imputation_mae"] == 3
    assert result["mape_percent"] is None
    np.testing.assert_allclose(result["nll"], .5 * np.log(2*np.pi) + 2)


def test_nonempty_training_faults_and_dropout():
    from rgdgl.data import fault_mask, TrafficWindows
    for family in ("random", "block", "persistent", "mixed"):
        available=np.zeros((2,3,1),dtype=bool);available[0,1,0]=True
        hidden=fault_mask(available,family,.1,13,nonempty=True)
        assert (hidden & available).sum()==1 and not (hidden & ~available).any()
    model=ReliabilityGraph(np.zeros((3,3)),width=4,horizon=1)
    assert model.dropout.p==.1
    x=torch.randn(2,3,3,1);m=torch.ones_like(x);clock=torch.randn(2,3,2)
    model.train();torch.manual_seed(1);a=model(x,m,clock)[1]
    torch.manual_seed(2);b=model(x,m,clock)[1]
    assert not torch.equal(a,b)
    model.eval();torch.testing.assert_close(model(x,m,clock)[1],model(x,m,clock)[1],atol=0,rtol=0)
    data=dict(x=np.zeros((10,3,1)),available=np.zeros((10,3,1),bool),clock=np.zeros(10))
    data["available"][7:]=True
    dataset=TrafficWindows(data,np.zeros_like(data["available"]),window=2,horizon=1,require_input=True)
    assert dataset.origins==[8,9]
