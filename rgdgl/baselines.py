"""Adapters to named upstream traffic models under one missingness protocol."""
import ast
import contextlib
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType,SimpleNamespace,MethodType
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.impute import KNNImputer


@contextlib.contextmanager
def upstream(path):
    """Keep historical projects' unqualified package names inside the loader."""
    prefixes={"model","models","utils","data","lib","modeling"}
    saved={key:value for key,value in sys.modules.items() if key.split('.')[0] in prefixes}
    for key in saved:del sys.modules[key]
    sys.path.insert(0,str(Path(path).resolve()))
    try:yield
    finally:
        sys.path.pop(0)
        for key in list(sys.modules):
            if key.split('.')[0] in prefixes:del sys.modules[key]
        sys.modules.update(saved)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def definitions(path,names,namespace):
    """Load exact definitions without executing a research script's CLI/training."""
    tree=ast.parse(Path(path).read_text(encoding="utf-8-sig"),filename=str(path))
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes}!=set(names):raise ValueError(f"Missing original definitions in {path}")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def modern_convolutions(model):
    """Old GWNet/GCN-M tuple kernels represent Conv2d, not modern Conv1d."""
    for name,child in list(model.named_children()):
        if isinstance(child,nn.Conv1d) and child.weight.ndim==4:
            replacement=nn.Conv2d(child.in_channels,child.out_channels,child.kernel_size,child.stride,
                                  child.padding,child.dilation,child.groups,child.bias is not None)
            replacement.load_state_dict(child.state_dict());setattr(model,name,replacement)
        else:modern_convolutions(child)


def local_feature_batch(model,inputs):
    """Preserve the batch axis while calling GCN-M's original local step."""
    return torch.stack([model.step(*(inputs[:,feature,step,:] for feature in range(8))) for step in range(inputs.shape[2])],dim=1)


def adjacency(distance):
    finite=distance[np.isfinite(distance)&(distance>0)]
    scale=float(np.std(finite)) if len(finite) else 1.
    matrix=np.exp(-np.square(distance/max(scale,1e-6))).astype('f4')
    np.fill_diagonal(matrix,1)
    return matrix


class NativeForecast(nn.Module):
    """Actual upstream network plus the explicitly named common output adapter."""
    def __init__(self,args,metadata,training):
        super().__init__();self.variant=args.variant;self.args=args
        self.forecaster=args.forecaster;self.imputer=args.imputer;self.objective_name=args.objective
        self.width,self.horizon=args.width,args.horizon
        distance=metadata['distance'];self.nodes=len(distance)
        channels=metadata['mean'].size
        if channels!=1:raise ValueError("Registered upstream traffic adapters require one speed channel")
        graph=adjacency(distance)
        self.register_buffer('graph',torch.tensor(graph))
        self.register_buffer('distance',torch.tensor(distance,dtype=torch.float32))
        forward=graph/graph.sum(1,keepdims=True).clip(1e-8)
        backward=graph.T/graph.T.sum(1,keepdims=True).clip(1e-8)
        self.register_buffer('forward_graph',torch.tensor(forward))
        self.register_buffer('backward_graph',torch.tensor(backward))
        source=training['x'][...,0];available=training['available'][...,0]
        count=available.sum(0);means=np.divide((source*available).sum(0),count,out=np.zeros(self.nodes),where=count>0)
        self.register_buffer('source_mean',torch.tensor(means,dtype=torch.float32))
        self.register_buffer('source_values',torch.tensor(np.where(available,source,np.nan),dtype=torch.float32))
        self.last_auxiliary=torch.tensor(0.)
        self.imputation_model=None
        if self.imputer in {'brits','grin'}:
            with upstream('.external/GRIN'):
                module=importlib.import_module('lib.nn.models.'+('brits' if self.imputer=='brits' else 'grin'))
                self.imputation_model=module.BRITSNet(self.nodes,args.width) if self.imputer=='brits' else module.GRINet(graph,1,args.width,args.width,args.dropout,impute_only_holes=False)
        elif self.imputer=='saits':
            with upstream('.external/SAITS'):
                cls=importlib.import_module('modeling.saits').SAITS
                self.imputation_model=cls(1,1,args.window,self.nodes,args.width,2*args.width,1,args.width,args.width,args.dropout,
                    input_with_mask=True,param_sharing_strategy='between_group',MIT=True,device=args.device,diagonal_attention_mask=True)
        elif self.imputer=='maginet':
            with upstream('.external/MagiNet'):
                module=importlib.import_module('models.model')
                names=['Learnable_Missing_Encoding','PositionalEmbedding','Temporal_Positional_Embedding','Adaptive_Missing_Spatial_Temporal_Encoder','MagiNet']
                namespace=dict(torch=torch,nn=nn,F=F,np=np,math=__import__('math'),make_model=module.make_model)
                cls=definitions(Path('.external/MagiNet/main.py'),names,namespace)['MagiNet']
                self.imputation_model=cls(args.device,self.nodes,args.window,1,args.width,1,2,1,args.width,graph)
                # Upstream Chebyshev tensors are plain attributes computed by
                # ARPACK. Preserve the actual fitted basis in the checkpoint.
                for part in self.imputation_model.modules():
                    if hasattr(part,'cheb_polynomials'):
                        for index,basis in enumerate(part.cheb_polynomials):part.register_buffer(f'graph_basis_{index}',basis.clone())
                        part.cheb_polynomials=[getattr(part,f'graph_basis_{index}') for index in range(len(part.cheb_polynomials))]
        if self.imputer in {'brits','grin','saits','maginet'} and self.forecaster!='agcrn':
            raise ValueError('Dedicated imputers must use the common AGCRN forecaster')
        if self.forecaster in {'agcrn','agcrn_mask'}:
            with upstream('.external/AGCRN'):
                cls=importlib.import_module('model.AGCRN').AGCRN
                self.network=cls(SimpleNamespace(num_nodes=self.nodes,input_dim=2 if self.forecaster=='agcrn_mask' else 1,rnn_units=args.width,
                    output_dim=1,horizon=args.horizon,num_layers=1,default_graph=True,embed_dim=10,cheb_k=args.diffusion+1))
            for parameter in self.network.parameters():
                if parameter.dim()>1:nn.init.xavier_uniform_(parameter)
                else:nn.init.uniform_(parameter)
        elif self.forecaster=='graphwavenet':
            with upstream('.external/Graph-WaveNet'):
                self.network=importlib.import_module('model').gwnet(args.device,self.nodes,dropout=args.dropout,supports=[self.forward_graph.to(args.device),self.backward_graph.to(args.device)],
                    in_dim=1,out_dim=args.horizon,residual_channels=args.width,dilation_channels=args.width,skip_channels=4*args.width,end_channels=8*args.width)
            modern_convolutions(self.network)
        elif self.forecaster=='dcrnn':
            with upstream('.external/DCRNN-source'),patch('torch.cuda.is_available',return_value=args.device.startswith('cuda')):
                # The PyTorch port imports TensorFlow for unused data utilities.
                # Load its exact graph-operator definitions without that import.
                utility=ModuleType('lib.utils');namespace=dict(np=np,sp=sp,linalg=linalg)
                names=['calculate_normalized_laplacian','calculate_random_walk_matrix','calculate_reverse_random_walk_matrix','calculate_scaled_laplacian']
                definitions(Path('.external/DCRNN-source/lib/utils.py'),names,namespace);utility.__dict__.update(namespace)
                package=ModuleType('lib');package.utils=utility;sys.modules['lib']=package;sys.modules['lib.utils']=utility
                cls=importlib.import_module('model.pytorch.dcrnn_model').DCRNNModel
                self.network=cls(graph,logging.getLogger('rgdgl.dcrnn'),num_nodes=self.nodes,input_dim=1,output_dim=1,rnn_units=args.width,
                    seq_len=args.window,horizon=args.horizon,num_rnn_layers=1,max_diffusion_step=args.diffusion,filter_type='dual_random_walk',use_curriculum_learning=False).to(args.device)
                # The original cell creates trainable matrices lazily.
                self.network(torch.zeros(args.window,2,self.nodes,device=args.device))
        elif self.forecaster=='uignn':
            module=load(Path('.external/UIGNN/structure_full.py'),'rgdgl_native_uignn')
            self.network=module.IGNNK(args.window,args.width,args.diffusion,forecasting=True,EDL=args.objective=='evidential')
            self.network.lin_out=nn.Linear(args.window,args.horizon)
            if args.objective=='evidential':self.evidential_loss=load(Path('.external/UIGNN/EDL_loss.py'),'rgdgl_native_edl').EvidentialLossSumOfSquares()
        elif self.forecaster=='gcnm':
            if self.nodes<20:raise ValueError('Original GCN-M dynamic top-k needs at least 20 sensors')
            with upstream('.external/GCN-M'),patch('torch.cuda.is_available',return_value=args.device.startswith('cuda')):
                graph_module=importlib.import_module('models.gnn');memory=importlib.import_module('models.memoryModule')
                namespace=dict(torch=torch,nn=nn,F=F,LocalFeatureModule=memory.LocalFeatureModule,MemoryModule=memory.MemoryModule)
                namespace.update({name:getattr(graph_module,name) for name in ['gcn_gwnet','gcn_gcnm_dynamic']})
                cls=definitions(Path('.external/GCN-M/models/model.py'),['GCNMdynamic'],namespace)['GCNMdynamic']
                self.network=cls(args.device,self.nodes,args.dropout,supports=[self.forward_graph.to(args.device),self.backward_graph.to(args.device)],in_dim=1,out_dim=args.horizon,
                    residual_channels=args.width,dilation_channels=args.width,skip_channels=4*args.width,end_channels=8*args.width)
            modern_convolutions(self.network)
            self.network.local_feature_model.forward=MethodType(local_feature_batch,self.network.local_feature_model)
        else:raise ValueError(f'Unknown native forecaster {self.forecaster}')
        if args.objective=='gaussian':self.adapter=nn.Linear(args.horizon,2*args.horizon)
        elif args.objective=='evidential' and args.forecaster!='uignn':raise ValueError('Evidential objective is native to UIGNN')
        self.knn=KNNImputer(n_neighbors=5,keep_empty_features=True).fit(self.source_values.numpy()) if self.imputer=='knn' else None

    def load_state_dict(self,*args,**kwargs):
        result=super().load_state_dict(*args,**kwargs)
        if self.imputer=='knn':self.knn=KNNImputer(n_neighbors=5,keep_empty_features=True).fit(self.source_values.detach().cpu().numpy())
        return result

    def fill(self,x,mask,clock):
        observed=x*mask;zero=x.sum()*0;self.last_auxiliary=zero
        if self.imputer=='mean':filled=torch.where(mask.bool(),x,self.source_mean[None,None,:,None])
        elif self.imputer=='forward':
            previous=self.source_mean[None,:,None].expand(len(x),-1,-1);rows=[]
            for step in range(x.shape[1]):
                previous=torch.where(mask[:,step].bool(),x[:,step],previous);rows.append(previous)
            filled=torch.stack(rows,1)
        elif self.imputer=='knn':
            values=torch.where(mask.bool(),x,torch.nan).detach().cpu().numpy().reshape(-1,self.nodes)
            filled=torch.tensor(self.knn.transform(values),device=x.device,dtype=x.dtype).reshape_as(x)
        elif self.imputer=='brits':
            output,imputations,predictions=self.imputation_model(observed[...,0],mask[...,0])
            self.last_auxiliary=((predictions-x[...,0][None]).abs()*mask[...,0][None]).sum()/mask.sum().clamp_min(1)/len(predictions)+.1*(imputations[0]-imputations[1]).abs().mean()
            filled=output[...,None]
        elif self.imputer=='grin':
            output=self.imputation_model(observed,mask.bool())
            if isinstance(output,tuple):
                filled,predictions=output
                self.last_auxiliary=((predictions-x[None]).abs()*mask[None]).sum()/mask.sum().clamp_min(1)/len(predictions)
            else:filled=output
        elif self.imputer=='saits':
            inputs=dict(X=observed[...,0],missing_mask=mask[...,0])
            filled,predictions=self.imputation_model.impute(inputs);filled=filled[...,None]
            # ORT on genuinely visible inputs; MIT is charged by objective below.
            self.last_auxiliary=sum(((value-x[...,0]).abs()*mask[...,0]).sum()/mask.sum().clamp_min(1) for value in predictions)/3
        elif self.imputer=='maginet':
            time=clock['time'] if isinstance(clock,dict) else clock
            position=(torch.atan2(time[...,0],time[...,1])/(2*torch.pi)%1*288).round()%288
            native_x=torch.cat([observed.permute(0,2,3,1),position[:,None,None].expand(-1,self.nodes,-1,-1)],2)
            native_mask=torch.cat([mask.permute(0,2,3,1),torch.ones_like(native_x[:,:,:1])],2)
            filled=self.imputation_model(native_x,native_mask).permute(0,3,1,2)
        else:raise ValueError(f'Unknown imputer {self.imputer}')
        return torch.where(mask.bool(),x,filled)

    def gcnm_inputs(self,x,mask,clock):
        raw=x[...,0];visible=mask[...,0]
        last=raw.new_zeros(len(x),self.nodes);age=torch.zeros_like(last);rows=[]
        distance=self.distance
        neighbors=distance.argsort(-1)[:,:min(5,self.nodes)]
        temporal_mean=(raw*visible).sum(1)/visible.sum(1).clamp_min(1)
        for step in range(x.shape[1]):
            value,seen=raw[:,step],visible[:,step]
            nearest=distance[None].expand(len(x),-1,-1).masked_fill(~seen[:,None].bool(),torch.inf).argmin(-1)
            any_visible=seen.any(-1,keepdim=True)
            closest=torch.where(any_visible,torch.gather(value,1,nearest),0)
            neighbor_values=value[:,neighbors];neighbor_mask=seen[:,neighbors]
            spatial_mean=(neighbor_values*neighbor_mask).sum(-1)/neighbor_mask.sum(-1).clamp_min(1)
            delta_s=torch.gather(distance[None].expand(len(x),-1,-1),2,nearest[:,:,None]).squeeze(-1)
            delta_s=torch.where(any_visible,delta_s,0)
            delta_s=torch.nan_to_num(delta_s,posinf=float(distance[torch.isfinite(distance)].max())+1)
            rows.append(torch.stack([value*seen,seen,last,temporal_mean,age,closest,spatial_mean,delta_s],1))
            last=torch.where(seen.bool(),value,last)
            age=torch.where(seen.bool(),torch.full_like(age,5),age+5)
        features=torch.stack(rows,2)
        histories=[];source=torch.nan_to_num(self.source_values);length=self.args.window
        for index,origin in enumerate(clock['origin'].tolist()):
            start=int(origin)-length;end=int(origin)-1;reference=[]
            # Published nh=1, nd=2, nw=2, tau=12: a complete bank of
            # recent/day/week windows. The bank is frozen source-training data.
            endpoints=list(range(start-length,end-length+1))
            for period in (288,2016):
                for repetition in (1,2):
                    endpoints.extend(range(start-repetition*period-length//2,end-repetition*period-length+length//2+1))
            for endpoint in endpoints:
                left=endpoint-length+1;right=endpoint+1
                if left>=0 and right<=min(len(source),start):reference.append(source[left:right])
                else:reference.append((raw[index]*visible[index]).detach())
            histories.append(torch.stack(reference))
        return features,torch.stack(histories)

    def forward(self,x,mask,clock):
        self.last_auxiliary=x.sum()*0
        if self.forecaster in {"graphwavenet","gcnm"}:self.network.supports=[self.forward_graph,self.backward_graph]
        if self.forecaster=='gcnm':
            features,history=self.gcnm_inputs(x,mask,clock);raw=self.network(features,history)
            reconstruction=self.network.local_feature_model(features)[...,None]
        else:
            reconstruction=self.fill(x,mask,clock)
            if self.forecaster in {'agcrn','agcrn_mask'}:
                inputs=torch.cat([reconstruction,mask],-1) if self.forecaster=='agcrn_mask' else reconstruction
                raw=self.network(inputs,None)[...,0]
            elif self.forecaster=='dcrnn':
                inputs=reconstruction[...,0].transpose(0,1).reshape(x.shape[1],len(x),self.nodes)
                raw=self.network(inputs).transpose(0,1).reshape(len(x),self.horizon,self.nodes)
            elif self.forecaster=='graphwavenet':raw=self.network(reconstruction.permute(0,3,2,1))[...,-1]
            else:
                raw,evidence,recovery=self.network(x[...,0]*mask[...,0],self.forward_graph,self.backward_graph)
                self.last_recovery=recovery[...,None]
                reconstruction=self.last_recovery
                self.last_evidence=evidence
        mean=raw[...,None];variance=torch.ones_like(mean)
        if self.objective_name=='gaussian':
            adapted=self.adapter(raw.transpose(1,2)).transpose(1,2)
            correction,log_variance=adapted.chunk(2,dim=1);mean=(raw+correction)[...,None];variance=F.softplus(log_variance)[...,None]+1e-6
        elif self.objective_name=='evidential':
            a,b,l=self.last_evidence[:,-3:,:].unbind(1)
            variance=((b+.3)*(l+2)/((l+1)*a.clamp_min(1e-6)))[:,None,:,None].expand_as(mean)
        return reconstruction,mean,variance,self.last_auxiliary

    def objective(self,x,available,hidden,clock,y,target_available,imputation_weight=1,entropy_weight=.001):
        reconstruction,mean,variance,auxiliary=self(x,available*(1-hidden),clock)
        if self.objective_name=='point':error=(mean-y).square() if self.forecaster=='uignn' else (mean-y).abs()
        elif self.objective_name=='evidential':
            evidence=self.last_evidence[:,-3:,:]
            rows=[]
            for horizon in range(self.horizon):
                valid=target_available[:,horizon,:,0].bool()
                if not valid.any():continue
                predicted=torch.cat([mean[:,horizon,:,0][:,None],evidence],1).permute(1,0,2)[:,valid].reshape(1,4,-1)
                rows.append(self.evidential_loss(predicted,y[:,horizon,:,0][valid].reshape(1,-1)))
            error=torch.stack(rows).mean()
        else:error=.5*(torch.log(2*torch.pi*variance)+(mean-y).square()/variance)
        forecast=error if error.ndim==0 else (error*target_available).sum()/target_available.sum().clamp_min(1)
        reconstruction_mask=available*hidden
        reconstruction_loss=((reconstruction-x).abs()*reconstruction_mask).sum()/reconstruction_mask.sum().clamp_min(1)
        if self.forecaster=='gcnm':return forecast
        if self.forecaster=='uignn':return forecast+.5*((self.last_recovery-x).square()*available*(1-hidden)).sum()/(available*(1-hidden)).sum().clamp_min(1)
        return forecast+imputation_weight*(reconstruction_loss+auxiliary)
