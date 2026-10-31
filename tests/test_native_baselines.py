"""Real upstream-model functional checks; no traffic benchmark result claims."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rgdgl.baselines import NativeForecast
from rgdgl.model import loss
from rgdgl.cli import main,metrics


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root=tmp_path_factory.mktemp("rgdgl_native_cli");rng=np.random.default_rng(13)
    nodes=20;distance=np.abs(np.arange(nodes)[:,None]-np.arange(nodes)[None,:]).astype("f4")
    np.savez(root/"input.npz",values=rng.normal(30,3,(160,nodes,1)).astype("f4"),distance=distance)
    original=sys.argv
    sys.argv=["rgdgl","prepare","--input",str(root/"input.npz"),"--output",str(root/"prepared"),"--realizations","1"]
    try:main()
    finally:sys.argv=original
    return root/"prepared"


@pytest.mark.parametrize('forecaster,imputer,objective',[
 ('agcrn','mean','point'),('agcrn','forward','gaussian'),('agcrn','knn','point'),('agcrn_mask','mean','gaussian'),
 ('dcrnn','mean','point'),('graphwavenet','mean','gaussian'),('agcrn','brits','point'),('agcrn','saits','gaussian'),
 ('agcrn','grin','point'),('agcrn','maginet','point'),('gcnm','mean','gaussian'),('uignn','mean','point'),('uignn','mean','evidential')])
def test_original_baseline_update_reload(forecaster,imputer,objective,tmp_path,prepared,monkeypatch):
    paths={'agcrn':'AGCRN','agcrn_mask':'AGCRN','dcrnn':'DCRNN-source','graphwavenet':'Graph-WaveNet','gcnm':'GCN-M','uignn':'UIGNN'}
    if not Path('.external',paths[forecaster]).is_dir():pytest.skip('Clone documented original sources')
    if imputer in {'brits','grin','saits','maginet'}:
        source={'brits':'GRIN','grin':'GRIN','saits':'SAITS','maginet':'MagiNet'}[imputer]
        if not Path('.external',source).is_dir():pytest.skip('Clone documented original imputer')
    torch.set_num_threads(1);torch.manual_seed(13)
    nodes=20;distance=np.abs(np.arange(nodes)[:,None]-np.arange(nodes)[None,:]).astype('f4')
    args=argparse.Namespace(variant='full',forecaster=forecaster,imputer=imputer,objective=objective,width=8,horizon=2,window=12,diffusion=2,dropout=0.,device='cpu')
    rng=np.random.default_rng(13);training=dict(x=rng.normal(size=(36,nodes,1)).astype('f4'),available=np.ones((36,nodes,1),bool))
    metadata=dict(distance=distance,mean=np.zeros((1,1,1)),scale=np.ones((1,1,1)))
    model=NativeForecast(args,metadata,training)
    x=torch.randn(2,12,nodes,1);available=torch.ones_like(x);hidden=torch.zeros_like(x);hidden[:,3:7,2:5]=1
    clock=torch.rand(2,12,2)
    if forecaster=='gcnm':clock=dict(time=clock,origin=torch.tensor([24,36]))
    y=torch.randn(2,2,nodes,1)
    before={key:value.detach().clone() for key,value in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    objective_value=loss(model,x,available,hidden,clock,y,torch.ones_like(y))
    assert torch.isfinite(objective_value)
    objective_value.backward();optimizer.step()
    assert any(not torch.equal(value,before[key]) for key,value in model.named_parameters())
    model.eval();expected=model(x,available*(1-hidden),clock)[1]
    checkpoint=tmp_path/'baseline.pt';torch.save(model.state_dict(),checkpoint)
    restored=NativeForecast(args,metadata,dict(training,x=training["x"]+99) if imputer=="knn" else training);restored.load_state_dict(torch.load(checkpoint,weights_only=True));restored.eval()
    torch.testing.assert_close(restored(x,available*(1-hidden),clock)[1],expected,rtol=0,atol=0)
    changed=x.clone();changed[hidden.bool()]+=1000
    torch.testing.assert_close(restored(changed,available*(1-hidden),clock)[1],expected,rtol=0,atol=0)

    # Run the named model through the shared real train/evaluate CLI.
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['rgdgl','train','--data',str(prepared),'--output',str(output),'--forecaster',forecaster,'--imputer',imputer,'--objective',objective,'--window','12','--horizon','2','--width','8','--epochs','1','--batch-size','32','--seed','13'])
    main()
    monkeypatch.setattr(sys,'argv',['rgdgl','evaluate','--data',str(prepared),'--model',str(output/'model.pt'),'--output',str(output/'evaluation'),'--batch-size','32'])
    main()
    result=json.loads((output/'evaluation/metrics.json').read_text())
    if objective=='point':assert 'nll' not in result and 'intervals' not in result
    if objective=='evidential':assert result['distribution']=='native NIG Student-t'
    monkeypatch.setattr(sys,'argv',['rgdgl','plot','--metrics',str(output/'evaluation/metrics.json'),'--output',str(output/'plot')])
    main()
    if forecaster=='gcnm':
        from rgdgl.analysis import benchmark
        monkeypatch.setattr(sys,'argv',['benchmark','--data',str(prepared),'--model',str(output/'model.pt'),'--output',str(output/'batch_one.json'),'--warmup','0','--repeats','1'])
        benchmark()
        assert json.loads((output/'batch_one.json').read_text())['batch_one_latency_ms']>0
    audit=json.loads((output/'training_masks.json').read_text())
    assert all(row['hidden']>0 or row['family']=='clean' for row in audit)


def test_student_crps_against_absolute_moment_quadrature():
    from scipy.integrate import quad
    from scipy.stats import t
    values=dict(mean=np.zeros((1,1,1,1)),variance=np.ones((1,1,1,1)),target=np.zeros((1,1,1,1)),available=np.ones((1,1,1,1)),
                imputation=np.zeros((1,1,1,1)),past=np.zeros((1,1,1,1)),hidden=np.ones((1,1,1,1)),nig=np.array([[[2.],[.7],[1.]]]))
    actual,_=metrics(values,0,1,1,5)
    # alpha=3,beta=1,lambda=2 => Student-t(6, scale=sqrt(.5)).
    scale=np.sqrt(.5);df=6
    expected=2*quad(lambda z:t.cdf(z,df)**2,-np.inf,0)[0]*scale
    np.testing.assert_allclose(actual['crps'],expected,rtol=1e-6)
