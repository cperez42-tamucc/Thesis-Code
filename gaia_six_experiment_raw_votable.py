from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gpytorch
from svgpr import SVGPRegressor, DeviceConfig, OptimizerConfig, NoiseMeanInitConfig
from svgpr.models.latent import LatentGPModel
from svgpr.models.noise import DispGPModel
from svgpr.config.settings import NoiseConfig
from gaia_raw_votable import prepare_votable

LOG = logging.getLogger('gaia_benchmark')
COMPONENTS = ('z', 'R', 'phi')
ROOT2PI = math.log(2.0 * math.pi)
CHI2_95 = float(scipy.stats.chi2.ppf(0.95, df=3))
PAPER = dict(epochs=300, batch_ratio=100.0, mean_lr=1.0,
             dispersion_lr=0.1, inducing=1000, optimizer='SGD',
             mean_kernel='RBF', dispersion_kernel='RQ',
             variational='NaturalVariationalDistribution')


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_write(path: Path, obj):
    def safe(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, Path): return str(v)
        if isinstance(v, torch.Tensor): return v.detach().cpu().tolist()
        if isinstance(v, np.ndarray): return v.tolist()
        raise TypeError(type(v).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=safe, allow_nan=False))


@dataclass
class Data:
    X: np.ndarray  # N,3; physical Galactocentric coordinates
    Y: np.ndarray  # N,3; velocity km/s
    obs: np.ndarray  # N,3 (sigma) or N,3,3 (covariance)
    source: str
    truth_mu: Optional[np.ndarray] = None
    truth_cov: Optional[np.ndarray] = None

    def __len__(self): return len(self.Y)

    def covariance_rows(self, idx: np.ndarray) -> np.ndarray:
        c = self.obs[idx]
        if c.ndim == 3:
            return np.asarray(c, dtype=np.float32)
        out = np.zeros((len(idx), 3, 3), dtype=np.float32)
        out[:, (0, 1, 2), (0, 1, 2)] = c.astype(np.float32)**2
        return out


def is_raw_votable(path: Path) -> bool:
    name=path.name.lower()
    return name.endswith(('.vot', '.vot.gz', '.xml', '.xml.gz', '.votable'))


def get_raw_cache_path(path: Path, args) -> Path:
    return args.raw_cache if args.raw_cache is not None else path.with_name('gaia_from_raw_votable.npz')


def load_gaia(path: Path, args) -> Data:
    if not path.is_file():
        raise FileNotFoundError(f'Missing Gaia input: {path}. Download the actual TAP result file to the HPC path.')
    if is_raw_votable(path):
        path=prepare_votable(path,get_raw_cache_path(path,args),
                             min_snr=args.min_parallax_snr,max_ruwe=args.ruwe_max,
                             chunk_rows=args.raw_chunk_rows,force=args.force_raw_preprocess)
    LOG.info('Loading Galactocentric Gaia cache into host RAM: %s', path)
    with np.load(path, allow_pickle=False) as arr:
        LOG.info('Available NPZ arrays: %s', list(arr.files))
        missing = {'X', 'V'} - set(arr.files)
        if missing:
            raise ValueError(f'Missing NPZ fields: {missing}; expected X=[z,R,phi], V=[vz,vR,vphi]')
        X = np.asarray(arr['X'], dtype=np.float32)
        Y = np.asarray(arr['V'], dtype=np.float32)
        if 'Cobs' in arr.files:
            obs = np.asarray(arr['Cobs'], dtype=np.float32)
            kind = 'full or measured covariance'
        elif 'E' in arr.files:
            obs = np.asarray(arr['E'], dtype=np.float32)
            kind = 'E diagonal standard deviations; off-diagonal measurement covariances unavailable'
        else:
            raise ValueError('Missing uncertainties: expected Cobs[N,3,3] or E[N,3]; never fabricate Gaia uncertainties')
    if X.ndim != 2 or Y.shape != X.shape or X.shape[1] != 3:
        raise ValueError(f'Bad input shapes X={X.shape}, V={Y.shape}; expected [N,3]')
    if obs.shape not in [(len(X),3), (len(X),3,3)]:
        raise ValueError(f'Bad uncertainty shape: {obs.shape}')
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y)) and np.all(np.isfinite(obs))):
        raise ValueError('Nonfinite input data. Clean your preprocessed Gaia catalogue before running.')
    if obs.ndim == 2 and np.any(obs < 0): raise ValueError('Negative measurement sigma found')
    if obs.ndim == 3 and np.any(np.diagonal(obs,axis1=1,axis2=2) < 0):
        raise ValueError('Negative observed covariance diagonal found')
    LOG.info('Read %d stars; uncertainty source: %s', len(X), kind)
    return Data(X=X, Y=Y, obs=obs, source=f'{path} ({kind})')


def make_mock(n: int, seed: int, spatial_strength: float=1.0,
              real_z: Optional[np.ndarray]=None) -> Data:
    """A paper-INSPIRED mock: decreasing density with |z|, tanh dispersion,
    Gaussian perturbations around +/-0.4 kpc, small latent mean excursions.
    Extension adds non-axisymmetric 3D features + off-diagonal covariance.
    Does not claim to reproduce the unpublished exact Gaia-derived curves.
    """
    rng = np.random.default_rng(seed)
    if real_z is None:
        z = rng.laplace(0, .35, n)
        while (bad := np.abs(z) > 2.5).any():
            z[bad] = rng.laplace(0, .35, int(bad.sum()))
    else:
        z = rng.choice(real_z, size=n, replace=(n > len(real_z))).astype(np.float64)
    R = np.clip(rng.normal(8.277, .23, n), 7.4, 9.2)
    phi = rng.uniform(-.18, .18, n)
    q = R-8.277
    g1 = np.exp(-.5*((z+.4)/.13)**2)
    g2 = np.exp(-.5*((z-.4)/.13)**2)
    base = 2.5*np.sin(z/.4)*np.exp(-z*z/1.3) + 1.5*(g2-g1)
    mu = np.stack((base, -.9*base, .5*base), axis=1)
    mu += spatial_strength * np.stack((3.0*q+9*phi, 7.0*q*np.cos(3*phi)+4*np.sin(5*phi),
                                       -12*q+6*np.sin(4*phi)),axis=1)
    tanh = np.tanh(np.abs(z)/.9)
    s0 = 18+12*tanh-2.0*g1+2.5*g2
    s1 = 25+15*tanh+3.0*g1-2.0*g2
    s2 = 16+11*tanh-1.0*g1+2.0*g2
    sigma = np.stack((s0,s1,s2),axis=1)
    sigma *= 1+spatial_strength*np.stack((.06*q+.05*np.sin(6*phi),
                                         -.08*q+.07*np.cos(4*phi),
                                          .06*q-.04*np.sin(5*phi)),axis=1)
    ell = np.zeros((n,3,3),dtype=np.float64)
    ell[:,0,0]=sigma[:,0]
    ell[:,1,0]=spatial_strength*.16*sigma[:,1]*np.tanh(2*z)
    ell[:,1,1]=sigma[:,1]
    ell[:,2,0]=spatial_strength*.10*sigma[:,2]*np.sin(5*phi)
    ell[:,2,1]=spatial_strength*.12*sigma[:,2]*np.tanh(z/.5)
    ell[:,2,2]=sigma[:,2]
    C=np.einsum('nij,nkj->nik',ell,ell).astype(np.float32)
    raw_velocity=mu+np.einsum('nij,nj->ni',ell,rng.standard_normal((n,3)))
    # Hapitas mock errors are 10% of simulated velocity magnitude.
    err=np.maximum(.10*np.abs(raw_velocity),.05).astype(np.float32)
    Y=(raw_velocity+rng.standard_normal((n,3))*err).astype(np.float32)
    return Data(X=np.stack((z,R,phi),axis=1).astype(np.float32),Y=Y,obs=err,
                source='synthetic Hapitas-inspired profiles with new 3D extension',
                truth_mu=mu.astype(np.float32),truth_cov=C)


def fixed_split(n: int, seed: int, fraction: float) -> tuple[np.ndarray,np.ndarray]:
    perm=np.random.default_rng(seed).permutation(n)
    ntest=max(1,int(round(n*fraction)))
    return np.asarray(perm[ntest:],dtype=np.int64), np.asarray(perm[:ntest],dtype=np.int64)


def subset(data: Data, n: int, seed: int, geometry:bool=True) -> Data:
    if geometry:
        X=data.X
        wrapped_phi=np.arctan2(np.sin(X[:,2]),np.cos(X[:,2]))
        good=np.flatnonzero((np.abs(X[:,0])<2.5) &
                             (np.abs(X[:,1]-8.277)<.25) &
                             (np.abs(wrapped_phi)<np.deg2rad(3.0)))
        LOG.info('Hapitas geometric cuts select %d of %d available rows',len(good),len(data))
        if len(good)<100:
            raise ValueError('Not enough Gaia data survive Hapitas solar-cylinder geometry. Check coordinate frame/units or use --subset-selection random')
    else:
        good=np.arange(len(data),dtype=np.int64)
    if n>=len(good):
        LOG.warning('Requested %d Gaia subset stars, but only %d eligible. Using all eligible.',n,len(good))
        ii=good
    else:
        ii=np.random.default_rng(seed+301).choice(good,n,replace=False)
    return Data(data.X[ii],data.Y[ii],data.obs[ii],data.source+f' subset {n}',
                None if data.truth_mu is None else data.truth_mu[ii],
                None if data.truth_cov is None else data.truth_cov[ii])


def spacefill_init(x: np.ndarray, M: int, seed: int) -> np.ndarray:
    """Sobol anchors in clipped domain; exactly M anchors, identical across runs."""
    lo=np.quantile(x,[.005],axis=0)[0]
    hi=np.quantile(x,[.995],axis=0)[0]
    if np.any(hi-lo<1e-5): raise ValueError('A spatial dimension has negligible dynamic range')
    sob=torch.quasirandom.SobolEngine(3,scramble=True,seed=seed)
    return (lo+(hi-lo)*sob.draw(M).numpy()).astype(np.float32)


def transform_spatial(x: torch.Tensor, loc: torch.Tensor,
                      scale: torch.Tensor) -> torch.Tensor:
    """Three physical coords mapped to Euclidean features with periodic phi."""
    z=(x[:,0]-loc[0])/scale[0]
    r=(x[:,1]-loc[1])/scale[1]
    phi=x[:,2]  # radians; respect seam using sin/cos
    return torch.stack((z,r,torch.sin(phi),torch.cos(phi)),dim=-1)


def robust_scaler(data: Data, train: np.ndarray, seed: int):
    pick=np.random.default_rng(seed+21).choice(train,size=min(len(train),200000),replace=False)
    x=data.X[pick].astype(np.float64)
    y=data.Y[pick].astype(np.float64)
    xloc=np.median(x[:,:2],axis=0)
    xscale=np.maximum(np.std(x[:,:2],axis=0),.1)
    yloc=np.mean(y,axis=0)
    yscale=np.maximum(np.std(y,axis=0),1.0)
    return {'xloc':[float(xloc[0]),float(xloc[1]),0.],
            'xscale':[float(xscale[0]),float(xscale[1]),1.],
            'yloc':yloc.tolist(),'yscale':yscale.tolist()}


def normalize_data(x, y, c, norm: dict, device: torch.device, dtype: torch.dtype):
    xt=torch.as_tensor(x,device=device,dtype=dtype)
    yt=torch.as_tensor(y,device=device,dtype=dtype)
    ct=torch.as_tensor(c,device=device,dtype=dtype)
    yloc=torch.tensor(norm['yloc'],device=device,dtype=dtype)
    ys=torch.tensor(norm['yscale'],device=device,dtype=dtype)
    loc=torch.tensor(norm['xloc'],device=device,dtype=dtype)
    scale=torch.tensor(norm['xscale'],device=device,dtype=dtype)
    return (transform_spatial(xt,loc,scale), (yt-yloc)/ys,
            ct/(ys[None,:,None]*ys[None,None,:]))


class ContextInducingNet(nn.Module):
    """SGNP-*inspired* set encoder. New surrogate-training method, not SGNP."""
    def __init__(self, width: int=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(6,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.decoder=nn.Sequential(nn.Linear(width+3,width),nn.SiLU(),nn.Linear(width,3))

    def forward(self, xc: torch.Tensor, yc: torch.Tensor, anchors: torch.Tensor):
        h=self.encoder(torch.cat((xc,yc),dim=1)).mean(dim=0)
        offsets=torch.tanh(self.decoder(torch.cat((anchors,h.expand(len(anchors),-1)),dim=1)))
        return anchors+.30*offsets


def train_inducing_net(data: Data, train: np.ndarray, anchors: np.ndarray,
                       norm: dict, args, out: Path, device):
    seed_all(args.seed+150)
    net=ContextInducingNet(args.nn_width).to(device=device,dtype=torch.float32)
    opt=torch.optim.Adam(net.parameters(), lr=args.nn_lr)
    rng=np.random.default_rng(args.seed+150)
    a=torch.as_tensor(anchors,device=device,dtype=torch.float32)
    loc=torch.as_tensor(norm['xloc'],device=device,dtype=torch.float32)
    scale=torch.as_tensor(norm['xscale'],device=device,dtype=torch.float32)
    ys=torch.as_tensor(norm['yscale'],device=device,dtype=torch.float32)
    ym=torch.as_tensor(norm['yloc'],device=device,dtype=torch.float32)
    to_norm=lambda x: (x-loc)/scale
    a0=to_norm(a)
    history=[]
    for t in range(args.nn_steps):
        idx=rng.choice(train,size=2*args.nn_context,replace=len(train)<2*args.nn_context)
        X=torch.as_tensor(data.X[idx],device=device,dtype=torch.float32)
        Y=(torch.as_tensor(data.Y[idx],device=device,dtype=torch.float32)-ym)/ys
        xc,xv=to_norm(X[:args.nn_context]),to_norm(X[args.nn_context:])
        yc,yv=Y[:args.nn_context],Y[args.nn_context:]
        z=net(xc,yc,a0)
        # Context -> proposed inducing anchors -> query kernel smoother.
        dctx=torch.cdist(z,xc).square()
        assign=F.softmax(-dctx/args.nn_temperature,dim=1)
        vz=assign@yc
        dq=torch.cdist(xv,z).square()
        predict=F.softmax(-dq/args.nn_temperature,dim=1)@vz
        err=F.smooth_l1_loss(predict,yv)
        reg=.003*(z-a0).square().mean()
        loss=err+reg
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step()
        history.append(float(loss.detach().cpu()))
        if (t+1)%25==0: LOG.info('Neural initializer %d/%d loss %.5f',t+1,args.nn_steps,history[-1])
    # For fixed fair initialization, compute one canonical training-only context.
    pick=rng.choice(train,size=min(args.nn_context*4,len(train)),replace=False)
    x=torch.as_tensor(data.X[pick],device=device,dtype=torch.float32)
    y=(torch.as_tensor(data.Y[pick],device=device,dtype=torch.float32)-ym)/ys
    with torch.no_grad(): z=net(to_norm(x),y,a0)*scale+loc
    result=z.detach().cpu().numpy().astype(np.float32)
    torch.save({'network':net.cpu().state_dict(),'anchors':anchors,'history':history,
                'note':'SGNP-inspired DeepSets/surrogate pretraining, not faithful SGNP'},
               out/'neural_initializer.pt')
    json_write(out/'neural_training.json',dict(loss_first=history[0] if history else None,
                loss_last=history[-1] if history else None,steps=args.nn_steps))
    return result


class IndependentBatchGP(gpytorch.models.ApproximateGP):
    """A collection of independent latent functions for one shared joint likelihood.
    3 RBF mean GPs plus 6 RQ Cholesky-factor GPs = 9 sparse latent GPs.
    """
    def __init__(self, inducing: torch.Tensor, count:int, kernel:str, dist:str):
        batch=torch.Size([count])
        M=inducing.size(0)
        if dist=='natural':
            vd=gpytorch.variational.NaturalVariationalDistribution(M,batch_shape=batch)
        else:
            vd=gpytorch.variational.CholeskyVariationalDistribution(M,batch_shape=batch)
        vs=gpytorch.variational.VariationalStrategy(
            self,inducing.unsqueeze(0).expand(count,-1,-1).clone(),
            vd,learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module=gpytorch.means.ZeroMean(batch_shape=batch)
        if kernel=='rbf': base=gpytorch.kernels.RBFKernel(batch_shape=batch,ard_num_dims=4)
        else: base=gpytorch.kernels.RQKernel(batch_shape=batch,ard_num_dims=4)
        self.covar_module=gpytorch.kernels.ScaleKernel(base,batch_shape=batch)

    def forward(self,x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),self.covar_module(x))


class JointTensor(nn.Module):
    def __init__(self, inducing_features: torch.Tensor, dist:str):
        super().__init__()
        self.mean_gp=IndependentBatchGP(inducing_features,3,'rbf',dist)
        self.factor_gp=IndependentBatchGP(inducing_features,6,'rq',dist)

    @staticmethod
    def intrinsic_cov(h:torch.Tensor):
        """Unrestricted six GP functions -> symmetric positive definite Sigma.
        h order: log-like Cholesky diagonals 00,10,11,20,21,22.
        """
        b=h.shape[0]
        L=h.new_zeros((b,3,3))
        L[:,0,0]=F.softplus(h[:,0])+.03
        L[:,1,0]=h[:,1]
        L[:,1,1]=F.softplus(h[:,2])+.03
        L[:,2,0]=h[:,3]
        L[:,2,1]=h[:,4]
        L[:,2,2]=F.softplus(h[:,5])+.03
        return L@L.transpose(-1,-2)

    def forward(self,x):
        a=self.mean_gp(x)
        b=self.factor_gp(x)
        return (a.mean.transpose(0,1),a.variance.transpose(0,1).clamp_min(0),
                b.mean.transpose(0,1),b.variance.transpose(0,1).clamp_min(0))


def batch_neglog(y:torch.Tensor, mu:torch.Tensor, varmu:torch.Tensor, intrinsic:torch.Tensor, measured:torch.Tensor):
    total=intrinsic+measured
    eye=torch.eye(3,device=y.device,dtype=y.dtype)
    L=torch.linalg.cholesky(total+1e-4*eye)
    resid=(y-mu).unsqueeze(-1)
    solved=torch.cholesky_solve(resid,L)
    inv=torch.cholesky_inverse(L)
    sq=(resid*solved).sum(dim=(-1,-2))
    epistemic=(inv.diagonal(dim1=-2,dim2=-1)*varmu).sum(dim=-1)
    ld=2*torch.log(torch.diagonal(L,dim1=-2,dim2=-1)).sum(dim=-1)
    return .5*(sq+epistemic+ld+3*ROOT2PI)


def joint_train(model:JointTensor, data:Data, indices:np.ndarray, norm:dict, args, out:Path, device:torch.device, epochs: int):
    out.mkdir(exist_ok=True,parents=True)
    model.to(device).train()
    groups=[{'params':model.mean_gp.parameters(),'lr':args.mean_lr},
            {'params':model.factor_gp.parameters(),'lr':args.disp_lr}]
    opt=torch.optim.SGD(groups)
    n=len(indices)
    nominal=max(1,int(n/args.batch_ratio))
    actual=min(nominal,args.batch_cap) if args.batch_cap>0 else nominal
    rng=np.random.default_rng(args.seed+33)
    g=torch.Generator(device=device).manual_seed(args.seed+88)
    steps=max(1,math.ceil(args.batch_ratio))
    losses=[]; start=time.perf_counter()
    order=rng.permutation(indices) if actual<nominal else None
    position=0
    visited=0
    starting_epoch=0
    checkpoint=out/'checkpoint_latest.pt'
    if args.resume and checkpoint.is_file():
        checkpoint_state=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(checkpoint_state['model'])
        opt.load_state_dict(checkpoint_state['optimizer'])
        rng.bit_generator.state=checkpoint_state['rng_state']
        g.set_state(checkpoint_state['generator_state'].cpu())
        starting_epoch=int(checkpoint_state['epoch'])
        losses=list(checkpoint_state['losses'])
        order=checkpoint_state['order']
        position=int(checkpoint_state['position'])
        visited=int(checkpoint_state['visited'])
        LOG.info('RESUME: continuing %s from completed epoch %d',out,starting_epoch)
    for epoch in range(starting_epoch,epochs):
        if order is None: shuffled=rng.permutation(indices)
        running=0.0;observed=0
        for k in range(steps):
            if order is None:
                begin=k*actual
                ids=shuffled[begin:min(begin+actual,n)]
                if len(ids)==0:break
            else:
                if position+actual<=n:
                    ids=order[position:position+actual]
                    position+=actual
                else:
                    part=order[position:]
                    order=rng.permutation(indices)
                    need=actual-len(part)
                    ids=np.concatenate((part,order[:need]))
                    position=need
                visited+=len(ids)
            x,y,c=normalize_data(data.X[ids],data.Y[ids],data.covariance_rows(ids),
                                 norm,device,args.torch_dtype)
            opt.zero_grad(set_to_none=True)
            with gpytorch.settings.cholesky_jitter(float_value=1e-4,double_value=1e-5):
                mu,vm,h,vh=model(x)
                nll=0
                for j in range(args.disp_mc):
                    eps=torch.randn(h.shape,device=device,dtype=h.dtype,generator=g)
                    covariance=model.intrinsic_cov(h+eps*torch.sqrt(vh+1e-6))
                    nll=nll+batch_neglog(y,mu,vm,covariance,c).mean()/args.disp_mc
                kl=(model.mean_gp.variational_strategy.kl_divergence().sum()
                    +model.factor_gp.variational_strategy.kl_divergence().sum())/n
                loss=nll+kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Joint objective non-finite at epoch {epoch+1},batch {k+1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=100.0)
            opt.step()
            running+=float(loss.detach())*len(ids);observed+=len(ids)
        losses.append(running/max(observed,1))
        if epoch==0 or (epoch+1)%max(1,epochs//1)==0:
            LOG.info('epoch %d/%d  ELBO-loss %.5f  elapsed %.1f min',
                     epoch+1,epochs,losses[-1],(time.perf_counter()-start)/60)
        if args.checkpoint_every>0 and (epoch+1)%args.checkpoint_every==0:
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                        'epoch':epoch+1,'norm':norm,'losses':losses,
                        'rng_state':rng.bit_generator.state,'generator_state':g.get_state(),
                        'order':order,'position':position,'visited':visited},checkpoint)
    seconds=time.perf_counter()-start
    return dict(epochs=epochs,steps_per_epoch=steps,batch_ratio_nominal=args.batch_ratio,
                batch_nominal=nominal,batch_actual=actual,
                complete_data_cycles_lower_bound=(steps*epochs*actual)//n,
                all_rows_seen_at_least_once=(order is None or visited>=n),
                batch_capped=bool(actual<nominal),
                epochs_as_in_svgpr_package=True,total_updates=steps*epochs,
                loss=losses,training_seconds=seconds)


def predict_joint(model:JointTensor, x:np.ndarray, norm:dict, device, dtype, mc:int=4):
    model.eval()
    B=len(x)
    # no observation covariance in intrinsic prediction
    empty=np.zeros((B,3,3),dtype=np.float32)
    a,_,_=normalize_data(x,np.zeros((B,3),dtype=np.float32),empty,norm,device,dtype)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(),\
         gpytorch.settings.cholesky_jitter(float_value=1e-4,double_value=1e-5):
        mu,vm,h,vh=model(a)
        # Quadrature-by-MC used only for intrinsic tensor mean; fixed seed
        # so fair across models, without contaminating training RNG.
        generator=torch.Generator(device=device).manual_seed(987654)
        cov=sum(model.intrinsic_cov(h+torch.randn(h.shape,device=device,dtype=dtype,
                                                    generator=generator)*torch.sqrt(vh+1e-7))
                for _ in range(mc))/mc
        ys=torch.tensor(norm['yscale'],device=device,dtype=dtype)
        ym=torch.tensor(norm['yloc'],device=device,dtype=dtype)
        mu=mu*ys+ym
        vm=vm*ys.square()
        cov=cov*(ys[None,:,None]*ys[None,None,:])
    return mu.detach().cpu().numpy(),vm.detach().cpu().numpy(),cov.detach().cpu().numpy()



class BatchedNoiseSVGPRegressor(SVGPRegressor):
    def _prepare_training(self):
        super()._prepare_training()
        batch_size=max(1,int(self.data_x.size(0)/self.optimizer_config.batch_ratio))
        dataset=TensorDataset(self.data_x,self.data_y,self.data_y_err)
        self.train_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

    def _run_training_loop(self):
        for ep in range(int(self.optimizer_config.n_epochs)):
            last=None
            for xb,yb,eb in self.train_loader:
                # FixedGaussianNoise honours a batch-shaped buffer; otherwise
                # its length-N original silently reduces to zero for minibatches.
                self.likelihood.noise_covar.noise=eb.contiguous()
                self.optimizer.zero_grad(set_to_none=True)
                latent=self.latent_model(xb)
                loss=-self.mll(latent,yb,inputs=xb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Hapitas package negative ELBO is nonfinite at epoch {ep+1}')
                loss.backward()
                self.optimizer.step()
                last=float(loss.detach())
            if ep==0 or (ep+1)%max(1,int(self.optimizer_config.n_epochs)//1)==0:
                LOG.info('Hapitas SVGPR epoch %d/%d, loss=%s',ep+1,
                         self.optimizer_config.n_epochs,last)


def baseline_fit(data:Data, train:np.ndarray, args, out:Path, device):
    seed_all(args.seed+33)
    models=[]; secs=[]
    # Package imports set torch default dtype to float64.
    x=torch.as_tensor(data.X[train,0],dtype=torch.float64)
    M=args.inducing
    init=torch.as_tensor(np.quantile(data.X[train,0],np.linspace(.001,.999,M)),dtype=torch.float64)
    for comp in range(3):
        yy=torch.as_tensor(data.Y[train,comp],dtype=torch.float64)
        obs_cov=data.covariance_rows(train)[:,comp,comp]
        sigma=np.sqrt(np.maximum(obs_cov,1e-10))
        # The released regressor divides passed error by y.std but passes the
        # result to FixedGaussianNoise as VARIANCE, not sigma; precompensate.
        ystd=yy.std().item()
        upstream_argument=torch.as_tensor(obs_cov/ystd,dtype=torch.float64)
        config=OptimizerConfig(batch_ratio=args.batch_ratio,
                              latent_lr=args.mean_lr,noise_lr=args.disp_lr,
                              n_epochs=args.epochs,optimizer_class=torch.optim.SGD)
        reg=BatchedNoiseSVGPRegressor(data_x=x,data_y=yy,
            latent_model=LatentGPModel(init.clone()).double(),
            noise_model=DispGPModel(init.clone()).double(),
            y_err=upstream_argument,
            optimizer_config=config,device_config=DeviceConfig(device=device),
            noise_init_config=NoiseMeanInitConfig(enabled=True,
                settings=NoiseConfig(bins=.1,bin_range=(-2.5,2.5),bin_size=True,
                                     debug=False)))
        LOG.info('Hapitas package component %s start (N=%d,M=%d,epochs=%d)',
                 COMPONENTS[comp],len(x),M,args.epochs)
        start=time.perf_counter()
        reg.fit()
        secs.append(time.perf_counter()-start)
        # Save provenance and model state, not entire training observations.
        torch.save(dict(latent=reg.latent_model.cpu().state_dict(),
                        noise=reg.noise_model.cpu().state_dict(),
                        y_mean=float(reg.data_y_mean),y_std=float(reg.data_y_std),
                        component=COMPONENTS[comp]),out/f'hapitas_{COMPONENTS[comp]}.pt')
        reg.latent_model.to(device)
        reg.noise_model.to(device)
        models.append(reg)
        LOG.info('Finished component %s in %.1f min',COMPONENTS[comp],secs[-1]/60)
    return models,dict(epochs=args.epochs,nominal_updates_per_component=
                        args.epochs*math.ceil(args.batch_ratio),
                        training_seconds_by_component=secs,
                        caveat='Released package executes 300 epochs x ~100 minibatches, not just 300 updates.')


def predict_baseline(models,x:np.ndarray,device):
    B=len(x)
    mu=np.empty((B,3),dtype=np.float64)
    varmu=np.empty((B,3),dtype=np.float64)
    C=np.zeros((B,3,3),dtype=np.float64)
    z=torch.as_tensor(x[:,0],dtype=torch.float64,device=device)
    for j,m in enumerate(models):
        with torch.no_grad():
            mm,lo,hi,sigma=m.predict(z)
        mu[:,j]=mm.detach().cpu().numpy()
        varmu[:,j]=((hi-lo)/4).detach().cpu().numpy()**2
        C[:,j,j]=sigma.detach().cpu().numpy()**2
    return mu,varmu,C


class EvalAccumulator:
    def __init__(self):
        self.n=0
        self.sq=np.zeros(3);self.absolute=np.zeros(3)
        self.nlpd=0.;self.cover=0.;self.marginal=np.zeros(3)
        self.latent_sq=np.zeros(3);self.disp_sq=np.zeros(3)
        self.tensor_sq=0.;self.tensor_offdiag_sq=0.;self.latent_cover=np.zeros(3)

    def add(self, y, pred, epistemic, intrinsic, meas, truthmu=None,truthcov=None):
        N=len(y);self.n+=N
        residual=y-pred
        self.sq+=(residual**2).sum(axis=0)
        self.absolute+=np.abs(residual).sum(axis=0)
        total=intrinsic+meas.copy()
        total[:,range(3),range(3)]+=epistemic
        total[:,range(3),range(3)]+=1e-5
        try:
            L=np.linalg.cholesky(total)
        except np.linalg.LinAlgError:
            total[:,range(3),range(3)]+=1e-2
            L=np.linalg.cholesky(total)
        whiten=np.linalg.solve(L,residual[...,None]).squeeze(-1)
        mahal=(whiten**2).sum(axis=1)
        ln_det=2*np.log(np.diagonal(L,axis1=1,axis2=2)).sum(axis=1)
        self.nlpd+=float((.5*(mahal+ln_det+3*ROOT2PI)).sum())
        self.cover+=float((mahal<=CHI2_95).sum())
        self.marginal+=(np.abs(residual)<=1.9599639845*np.sqrt(np.diagonal(total,axis1=1,axis2=2))).sum(axis=0)
        if truthmu is not None:
            self.latent_sq+=((pred-truthmu)**2).sum(axis=0)
            self.latent_cover+=(np.abs(pred-truthmu)<=1.9599639845*np.sqrt(np.maximum(epistemic,0))).sum(axis=0)
            true_sig=np.sqrt(np.diagonal(truthcov,axis1=1,axis2=2))
            pred_sig=np.sqrt(np.maximum(np.diagonal(intrinsic,axis1=1,axis2=2),1e-12))
            self.disp_sq+=((pred_sig-true_sig)**2).sum(axis=0)
            err=(intrinsic-truthcov)**2
            self.tensor_sq+=float(err.sum())
            self.tensor_offdiag_sq+=float((err[:,0,1]+err[:,0,2]+err[:,1,2]).sum())

    def finish(self, truth:bool):
        n=max(1,self.n)
        out={'n_evaluated':self.n,
             'observed_velocity_rmse_kms':(np.sqrt(self.sq/n)).tolist(),
             'observed_velocity_mae_kms':(self.absolute/n).tolist(),
             'observed_velocity_rmse_macro_kms':float(np.sqrt(self.sq.sum()/(3*n))),
             'observed_nlpd_per_star':self.nlpd/n,
             'observed_joint_95_region_coverage':self.cover/n,
             'observed_marginal_95_interval_coverage':(self.marginal/n).tolist(),
             'true_intrinsic_tensor_known':truth}
        if truth:
            out.update(latent_mean_mse_kms2=(self.latent_sq/n).tolist(),
               dispersion_sigma_mse_kms2=(self.disp_sq/n).tolist(),
               intrinsic_tensor_entry_mse_kms4=self.tensor_sq/(n*9),
               intrinsic_tensor_offdiag_mse_kms4=self.tensor_offdiag_sq/(n*3),
               true_latent_mean_95_region_coverage=(self.latent_cover/n).tolist())
        return out


def evaluate(data, idx, norm, args, output:Path, model=None, baseline=None, device=None):
    acc=EvalAccumulator();probe_scores=[];probe_indices=[];probe_coords=[]
    diagnostic=None
    for k in range(0,len(idx),args.eval_batch):
        ids=idx[k:k+args.eval_batch]
        if baseline is None:
            p,ep,ci=predict_joint(model,data.X[ids],norm,device,args.torch_dtype,args.predict_mc)
        else:
            p,ep,ci=predict_baseline(baseline,data.X[ids],device)
        c=data.covariance_rows(ids)
        y=data.Y[ids]
        if diagnostic is None:
            take=min(2500,len(ids))
            diagnostic=dict(x=data.X[ids[:take]],observed=y[:take],predicted=p[:take],
                            intrinsic_cov=ci[:take],
                            true_mean=None if data.truth_mu is None else data.truth_mu[ids[:take]],
                            true_intrinsic_cov=None if data.truth_cov is None else data.truth_cov[ids[:take]])
        acc.add(y,p,ep,ci,c,
                None if data.truth_mu is None else data.truth_mu[ids],
                None if data.truth_cov is None else data.truth_cov[ids])
        if len(probe_indices)<args.probe_max:
            total=ci+c.copy()
            total[:,range(3),range(3)]+=ep+1e-4
            try: L=np.linalg.cholesky(total)
            except np.linalg.LinAlgError:
                total[:,range(3),range(3)]+=1e-2
                L=np.linalg.cholesky(total)
            r=(y-p)[...,None]
            white=np.linalg.solve(L,r).squeeze(-1)
            score=.5*((white**2).sum(axis=1)+
                        2*np.log(np.diagonal(L,axis1=1,axis2=2)).sum(axis=1)+3*ROOT2PI)
            take=min(len(ids),args.probe_max-len(probe_indices))
            probe_scores.extend(score[:take].tolist())
            probe_indices.extend(ids[:take].tolist())
            probe_coords.extend(data.X[ids[:take]].tolist())
        if k==0 or (k//args.eval_batch)%max(1,50000//args.eval_batch)==0:
            LOG.info('Evaluated %s / %s stars',min(k+args.eval_batch,len(idx)),len(idx))
    met=acc.finish(data.truth_mu is not None)
    np.savez_compressed(output/'paired_probe.npz',indices=np.asarray(probe_indices,dtype=np.int64),
                        xyz=np.asarray(probe_coords,dtype=np.float32),
                        nlpd=np.asarray(probe_scores,dtype=np.float32))
    json_write(output/'metrics.json',met)
    if diagnostic is not None:
        np.savez_compressed(output/'diagnostic_predictions.npz',**{k:v for k,v in diagnostic.items() if v is not None})
        plot_diagnostics(output,diagnostic)
    return met


def plot_diagnostics(output:Path,d):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for j,ax in enumerate(axes):
        ax.scatter(d['observed'][:,j],d['predicted'][:,j],s=3,alpha=.2)
        q=np.quantile(np.r_[d['observed'][:,j],d['predicted'][:,j]],[.01,.99])
        ax.plot(q,q,linestyle='--',color='black',linewidth=1)
        ax.set(xlabel=f'observed v_{COMPONENTS[j]} (km/s)',
               ylabel='predicted mean (km/s)',title=COMPONENTS[j])
    fig.tight_layout();fig.savefig(output/'observed_vs_predicted.png',dpi=150);plt.close(fig)
    if d['true_mean'] is not None:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for j in range(3):
            axes[0,j].scatter(d['true_mean'][:,j],d['predicted'][:,j],s=4,alpha=.3)
            q=np.quantile(d['true_mean'][:,j],[.01,.99])
            axes[0,j].plot(q,q,color='black',linestyle='--',linewidth=1)
            axes[0,j].set(xlabel='true latent mean',ylabel='inferred mean',title=COMPONENTS[j])
            tru=np.sqrt(d['true_intrinsic_cov'][:,j,j])
            est=np.sqrt(np.maximum(d['intrinsic_cov'][:,j,j],0))
            axes[1,j].scatter(tru,est,s=4,alpha=.3)
            q=np.quantile(tru,[.01,.99])
            axes[1,j].plot(q,q,color='black',linestyle='--',linewidth=1)
            axes[1,j].set(xlabel='true intrinsic sigma',ylabel='inferred sigma')
        fig.tight_layout();fig.savefig(output/'truth_recovery.png',dpi=150);plt.close(fig)


def compare(a:Path,b:Path,out:Path,seed:int):
    m1=json.loads((a/'metrics.json').read_text())
    m2=json.loads((b/'metrics.json').read_text())
    sa=np.load(a/'paired_probe.npz');sb=np.load(b/'paired_probe.npz')
    if not np.array_equal(sa['indices'],sb['indices']):
        raise ValueError('Paired analysis refused: evaluation row identities differ')
    diff=sa['nlpd'].astype(np.float64)-sb['nlpd'].astype(np.float64)
    coords=sa['xyz'].astype(np.float64)
    if not np.array_equal(sa['xyz'],sb['xyz']):
        raise ValueError('Paired analysis refused: probe coordinates differ')
    cell=np.zeros(len(diff),dtype=np.int64)
    stride=1
    for axis in range(3):
        edges=np.quantile(coords[:,axis],[.25,.5,.75])
        cell+=stride*np.searchsorted(edges,coords[:,axis],side='right')
        stride*=4
    groups, inverse=np.unique(cell,return_inverse=True)
    counts=np.bincount(inverse).astype(np.float64)
    means=np.bincount(inverse,weights=diff)/counts
    rng=np.random.default_rng(seed+442)
    boots=np.empty(500)
    for j in range(500):
        draw=rng.integers(0,len(groups),len(groups))
        boots[j]=np.dot(counts[draw],means[draw])/np.sum(counts[draw])
    lo,hi=np.quantile(boots,[.025,.975])
    result=dict(first=a.name,second=b.name,positive_means_second_is_better=True,
        paired_probe_n=len(diff),mean_nlpd_improvement_second=float(diff.mean()),
        paired_bootstrap_95_CI=[float(lo),float(hi)],
        bootstrap_unit='3D spatial block (4 quantile bins per spatial axis)',
        n_spatial_blocks=int(len(groups)),
        evidence_second_better_on_nlpd=bool(lo>0),
        first_observed_nlpd=m1['observed_nlpd_per_star'],
        second_observed_nlpd=m2['observed_nlpd_per_star'],
        first_joint_95_coverage=m1['observed_joint_95_region_coverage'],
        second_joint_95_coverage=m2['observed_joint_95_region_coverage'])
    json_write(out,result)
    return result


def new_joint(anchors, norm, args, device):
    device_dtype=args.torch_dtype
    x=torch.as_tensor(anchors,device=device,dtype=device_dtype)
    loc=torch.tensor(norm['xloc'],device=device,dtype=device_dtype)
    scale=torch.tensor(norm['xscale'],device=device,dtype=device_dtype)
    inducing=transform_spatial(x,loc,scale)
    return JointTensor(inducing,args.variational).to(device=device,dtype=device_dtype)


def run_one(mode:str,data:Data,train:np.ndarray,test:np.ndarray,
            args,base:Path,device,norm:dict):
    root=base/mode;root.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed+33)
    init_idx=np.random.default_rng(args.seed+40).choice(train,size=min(100000,len(train)),replace=False)
    anchors=spacefill_init(data.X[init_idx],args.inducing,args.seed+4)
    start=time.perf_counter()
    if 'hapitas' in mode:
        baselines,training=baseline_fit(data,train,args,root,device)
        metrics=evaluate(data,test,norm,args,root,baseline=baselines,device=device)
        del baselines
    else:
        if 'neural' in mode:
            induce=train_inducing_net(data,train,anchors,norm,args,root,device)
        else:
            induce=anchors
        seed_all(args.seed+880)
        model=new_joint(induce,norm,args,device)
        training=joint_train(model,data,train,norm,args,root,device,args.epochs)
        torch.save({'model':model.cpu().state_dict(), 'norm':norm,
                    'inducing_initial_physical':induce,'hyperparameters':PAPER,
                    'architecture':'nine sparse GPs, joint three-output Gaussian likelihood, SPD Cholesky intrinsic tensor'},
                   root/'model.pt')
        model.to(device)
        metrics=evaluate(data,test,norm,args,root,model=model,device=device)
        del model
    seconds=time.perf_counter()-start
    config=dict(mode=mode,source=data.source,total_rows=len(data),training_rows=len(train),
                heldout_rows=len(test),seed=args.seed,gaia_validation_heldout=True,
                paper_nominal=PAPER,actual_epochs=args.epochs,
                actual_inducing=args.inducing,actual_mean_lr=args.mean_lr,
                actual_dispersion_lr=args.disp_lr,actual_batch_ratio=args.batch_ratio,
                neural_steps=args.nn_steps if 'neural' in mode else 0,
                actual_optimizer='SGD',full_test_evaluated=True,
                training=training,normalization=norm,wall_seconds=seconds)
    json_write(root/'run.json',config)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--mode',choices=['all','mock','subset','full','mock-hapitas','mock-neural',
                                    'subset-hapitas','subset-neural','full-neural','full-spacefill'],default='all')
    ap.add_argument('--gaia',type=Path,default=Path('/home/cperez85/neural tests/2cb451da-d64b-11f1-b71e-bc97e148b76b-O-result.vot'),
                     help='Raw Gaia TAP VOTable or a previously converted NPZ')
    ap.add_argument('--raw-cache',type=Path,default=None,
                     help='Raw VOTable derived cache path; never serves as source of truth')
    ap.add_argument('--prepare-gaia-only',action='store_true',
                     help='Convert raw Gaia VOTable to cache and exit, without any GP training')
    ap.add_argument('--force-raw-preprocess',action='store_true',
                     help='Force regeneration of derived cache')
    ap.add_argument('--min-parallax-snr',type=float,default=5.0)
    ap.add_argument('--ruwe-max',type=float,default=1.4)
    ap.add_argument('--raw-chunk-rows',type=int,default=65536)
    ap.add_argument('--output',type=Path,default=Path('six_experiment_outputs'))
    ap.add_argument('--subset-selection',choices=['hapitas-geometry','random'],default='hapitas-geometry')
    ap.add_argument('--subset-n',type=int,default=833808,
                    help='Processed Gaia subset size, paper used 833,808 AFTER its specific cuts')
    ap.add_argument('--mock-n',type=int,default=800000)
    ap.add_argument('--mock-spatial-strength',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=25)
    ap.add_argument('--test-fraction',type=float,default=.20)
    ap.add_argument('--inducing',type=int,default=1000)
    ap.add_argument('--epochs',type=int,default=300,
                    help='The released svgpr library treats 300 as full EPOCHS, not 300 steps')
    ap.add_argument('--batch-ratio',type=float,default=100.)
    ap.add_argument('--batch-cap',type=int,default=8192,
                    help='Joint GPU feasibility cap; for full Gaia this changes effective batch ratio')
    ap.add_argument('--mean-lr',type=float,default=1.)
    ap.add_argument('--disp-lr',type=float,default=.1)
    ap.add_argument('--variational',choices=['natural','cholesky'],default='natural')
    ap.add_argument('--disp-mc',type=int,default=2)
    ap.add_argument('--predict-mc',type=int,default=4)
    ap.add_argument('--nn-steps',type=int,default=150)
    ap.add_argument('--nn-context',type=int,default=192)
    ap.add_argument('--nn-width',type=int,default=64)
    ap.add_argument('--nn-lr',type=float,default=.001)
    ap.add_argument('--nn-temperature',type=float,default=.25)
    ap.add_argument('--eval-batch',type=int,default=2048)
    ap.add_argument('--probe-max',type=int,default=100000)
    ap.add_argument('--checkpoint-every',type=int,default=25)
    ap.add_argument('--resume',action='store_true',help='Resume checkpointed joint training from last completed epoch')
    ap.add_argument('--no-refit-full',action='store_true',help='Skip the final all-star fit after held-out full benchmark')
    ap.add_argument('--dtype',choices=['float32','float64'],default='float64')
    ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args()
    if not (0<a.test_fraction<1):ap.error('--test-fraction must be between 0 and 1')
    if a.inducing<4 or a.epochs<1 or a.batch_ratio<1:ap.error('Invalid inducing/epochs/batch ratio')
    a.torch_dtype=torch.float32 if a.dtype=='float32' else torch.float64
    torch.set_default_dtype(a.torch_dtype)
    a.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(a.output/'benchmark.log')])
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    seed_all(a.seed)
    LOG.info('Device %s; torch %s; gpytorch %s; dtype %s',device,torch.__version__,gpytorch.__version__,a.dtype)
    LOG.warning('Scientific constraints: SGNP-inspired initializer is a NEW extension, not exact SGNP; superiority must be measured.')
    LOG.warning('Full-Gaia sampling batch cap %d may differ from paper batch ratio 100. See run.json.',a.batch_cap)
    LOG.warning('Hapitas 300 steps versus released package 300 epochs is an upstream provenance mismatch.')
    if a.prepare_gaia_only:
        if not is_raw_votable(a.gaia):
            ap.error('--prepare-gaia-only requires --gaia pointing to the VOTable file')
        result=prepare_votable(a.gaia,get_raw_cache_path(a.gaia,a),
                               min_snr=a.min_parallax_snr,max_ruwe=a.ruwe_max,
                               chunk_rows=a.raw_chunk_rows,force=a.force_raw_preprocess)
        LOG.info('Prepared raw input as %s; ready for model experiments',result)
        return
    if a.mode=='all':modes=['mock-hapitas','mock-neural','subset-hapitas','subset-neural','full-neural','full-spacefill']
    elif a.mode=='mock':modes=['mock-hapitas','mock-neural']
    elif a.mode=='subset':modes=['subset-hapitas','subset-neural']
    elif a.mode=='full':modes=['full-neural','full-spacefill']
    else:modes=[a.mode]
    groups={p.split('-')[0] for p in modes}
    for group in ('mock','subset','full'):
        if group not in groups:continue
        if group=='mock':
            ref=None
            if a.gaia.exists():
                try:
                    mock_reference=get_raw_cache_path(a.gaia,a) if is_raw_votable(a.gaia) else a.gaia
                    with np.load(mock_reference,allow_pickle=False) as f:
                        if 'X' in f.files:
                            zx=np.asarray(f['X'][:,0],dtype=np.float32)
                            ref=zx[np.isfinite(zx)&(np.abs(zx)<2.5)]
                except Exception as e: LOG.warning('Gaia z sampling unavailable: %s; Laplace approximation used',e)
            d=make_mock(a.mock_n,a.seed,a.mock_spatial_strength,real_z=ref)
            del ref
        else:
            d=load_gaia(a.gaia,a)
            if group=='subset': d=subset(d,a.subset_n,a.seed,geometry=(a.subset_selection=='hapitas-geometry'))
        train,test=fixed_split(len(d),a.seed,a.test_fraction)
        norm=robust_scaler(d,train,a.seed)
        json_write(a.output/f'{group}_dataset.json',dict(n=len(d),train=len(train),
                   test=len(test),source=d.source,has_truth=d.truth_mu is not None,
                   split_seed=a.seed,spatial_strength=a.mock_spatial_strength if group=='mock' else None,
                   normalization=norm))
        for mode in modes:
            if not mode.startswith(group+'-'):continue
            run_one(mode,d,train,test,a,a.output,device,norm)
        left=a.output/('mock-hapitas' if group=='mock' else 'subset-hapitas' if group=='subset' else 'full-spacefill')
        right=a.output/('mock-neural' if group=='mock' else 'subset-neural' if group=='subset' else 'full-neural')
        if (left/'paired_probe.npz').exists() and (right/'paired_probe.npz').exists():
            result=compare(left,right,a.output/f'{group}_paired_comparison.json',a.seed)
            LOG.info('PAIRED %s difference %+0.5f 95%% CI %s',group,
                     result['mean_nlpd_improvement_second'],result['paired_bootstrap_95_CI'])
        if group=='full' and not a.no_refit_full:
            for method in ('neural','spacefill'):
                if f'full-{method}' not in modes:continue
                base=a.output/f'full-{method}'
                artifact=torch.load(base/'model.pt',map_location='cpu',weights_only=False)
                zz=artifact['inducing_initial_physical']
                fresh=new_joint(zz,norm,a,device)
                entire=np.arange(len(d),dtype=np.int64)
                stats=joint_train(fresh,d,entire,norm,a,base/'refit_all',device,a.epochs)
                (base/'refit_all').mkdir(exist_ok=True,parents=True)
                torch.save({'model':fresh.cpu().state_dict(),'norm':norm,
                           'all_catalogue_n':len(d),'fit_stats':stats},base/'refit_all'/'full_catalogue_final.pt')
                json_write(base/'refit_all'/'training.json',stats)
                del fresh
        del d,train,test
        gc.collect()
    LOG.info('Completed requested modes. See outputs/metrics.json and *_paired_comparison.json')


if __name__=='__main__':
    main()
