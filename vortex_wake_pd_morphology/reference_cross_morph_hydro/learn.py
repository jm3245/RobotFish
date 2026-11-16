"""Fresh WM, then equal-budget A/B/C behavioral cloning on the same episodes."""
import argparse, hashlib, json
import numpy as np
import h5py
import torch
from torch import nn
from fish_rl.shared_rl.model import mlp
from .config import RUN,TRAIN_N,DEVICE,SEEDS,PROTOCOL,write,sha,check_stop

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

class InteractionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=nn.Sequential(nn.Conv1d(36,128,5,padding=2),nn.ELU(),
            nn.Conv1d(128,128,5,padding=2),nn.ELU(),
            nn.Conv1d(128,128,5,padding=2),nn.ELU())
        self.output=nn.Linear(128,16)
    def forward(self,history):
        assert history.shape[-2:]==(50,36)
        return self.output(self.core(history.transpose(1,2)).mean(-1))

class World(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=InteractionEncoder()
        self.actions=nn.GRU(4,128,batch_first=True)
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(161,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,44)) for _ in range(3)])
    def forward(self,h,current,actions,z_mode='normal'):
        z=self.encoder(h)
        if z_mode=='zero':z=z*0
        if z_mode=='shuffle':z=z.roll(1,0)
        action,_=self.actions(actions)
        p=torch.stack([head(torch.cat((current,z,action[:,t-1]),-1)) for head,t in zip(self.heads,(10,25,50))],1)
        return p

class Policy(nn.Module):
    def __init__(self,kind,norm,seed,world_state=None):
        super().__init__();self.kind=kind
        for k in ('observation_mean','observation_std','history_mean','history_std'):
            self.register_buffer(k,torch.as_tensor(norm[k],dtype=torch.float32))
        torch.manual_seed(seed);base=mlp(17,4)
        torch.manual_seed(seed);self.actor=mlp(17 if kind=='A' else 33,4)
        with torch.no_grad():
            for i in (0,2,4,6):
                self.actor[i].weight.zero_()
                if i==0:self.actor[i].weight[:,:17].copy_(base[i].weight)
                else:self.actor[i].weight.copy_(base[i].weight)
                self.actor[i].bias.copy_(base[i].bias)
        if kind!='A':
            torch.manual_seed(seed+100);self.encoder=InteractionEncoder()
            if kind=='C':
                if world_state is None:raise ValueError('C requires this run fresh pretraining')
                self.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in world_state.items() if k.startswith('encoder.')})
                self.encoder.requires_grad_(False)
    def forward(self,observation,history,mode='normal'):
        x=(observation-self.observation_mean)/self.observation_std
        if self.kind!='A':
            z=self.encoder((history-self.history_mean)/self.history_std)
            if mode=='zero':z=z*0
            if mode=='shuffle':z=z.roll(1,0)
            x=torch.cat((x,z),-1)
        return self.actor(x).tanh()

def data():
    audit=json.loads((RUN/'dataset_audit.json').read_text());assert audit['passed']
    with h5py.File(RUN/'dataset.h5','r') as h:d={k:h[k][:] for k in h}
    assert set(np.unique(d['n']))==set(TRAIN_N)
    return d

def dataset_sha():return json.loads((RUN/'dataset_audit.json').read_text())['sha256']

def normalization(d):
    mask=d['split']==0;norm={}
    for key in ('observation','history','current','future_actions','labels'):
        ids=np.flatnonzero(mask);total=0;s=None;s2=None
        for start in range(0,len(ids),4096):
            x=d[key][ids[start:start+4096]].astype(np.float64)
            if key in ('history','future_actions'):x=x.reshape(-1,x.shape[-1])
            a=x.sum(0);b=(x*x).sum(0)
            s=a if s is None else s+a;s2=b if s2 is None else s2+b;total+=len(x)
        mean=s/total
        floor=1e-4 if key in ('history','labels') else .02
        std=np.sqrt(np.maximum(s2/total-mean*mean,0)).clip(floor)
        norm[key+'_mean']=mean.astype(np.float32);norm[key+'_std']=std.astype(np.float32)
    return norm

def tensor(d,key,ix):return torch.as_tensor(d[key][ix],device=DEVICE,dtype=torch.float32)
def normtensor(d,key,ix,n):return (tensor(d,key,ix)-n[key+'_mean'])/n[key+'_std']

def sampler(d,seed):
    rng=np.random.default_rng(seed)
    # Balance jointly over topology, morphology size and flow category. Each
    # method receives the exact same sampled indices for each seed/update.
    cells=[]
    for n in TRAIN_N:
        for m in range(12):
            for f in range(6):
                ids=np.flatnonzero((d['split']==0)&(d['n']==n)&(d['morph']==m)&(d['flow_kind']==f))
                if not len(ids):continue  # Explicitly report absent cells; never fabricate samples.
                cells.append(ids)
    assert cells
    while True:
        cells_chosen=rng.integers(len(cells),size=PROTOCOL['batch_size'])
        yield np.array([rng.choice(cells[i]) for i in cells_chosen])

def world_loss(model,d,ix,n,mode='normal'):
    pred=model(normtensor(d,'history',ix,n),normtensor(d,'current',ix,n),normtensor(d,'future_actions',ix,n),mode)
    target=normtensor(d,'labels',ix,n)
    hydro=(pred[:,:,:32]-target[:,:,:32]).square().mean()
    motion=(pred[:,:,32:]-target[:,:,32:]).square().mean()
    # True flow and morphology are diagnostics/stratification metadata only.
    # Even WM pretraining does not receive privileged flow labels.
    return hydro+motion

@torch.no_grad()
def score_world(model,d,ix,n,mode='normal'):
    losses=[];weights=[]
    for start in range(0,len(ix),512):
        batch=ix[start:start+512];losses.append(float(world_loss(model,d,batch,n,mode)));weights.append(len(batch))
    return float(np.average(losses,weights=weights))

@torch.no_grad()
def world_diagnostics(model,d,ix,n):
    accum=np.zeros((3,2,4),dtype=np.float64)
    count=0
    for start in range(0,len(ix),256):
        ids=ix[start:start+256]
        pred=model(normtensor(d,'history',ids,n),normtensor(d,'current',ids,n),
                   normtensor(d,'future_actions',ids,n))
        actual=tensor(d,'labels',ids)
        pred=pred*n['labels_std']+n['labels_mean']
        history=tensor(d,'history',ids)
        observation=tensor(d,'observation',ids)
        for j,dt in enumerate((.2,.5,1.)):
            persist=history[:,-1,:32]
            velocity=observation[:,2:5];angular=observation[:,5:8]
            straight=torch.cat((velocity*dt,angular*dt,velocity,angular),-1)
            baseline=torch.cat((persist,straight),-1)
            for k,value in enumerate((pred[:,j],baseline)):
                diff=(value-actual[:,j]).square()
                accum[j,k,0]+=float(diff[:,:32].sum())
                accum[j,k,1]+=float(diff[:,32:].sum())
                accum[j,k,2]+=float(diff[:,:32].reshape(-1,8,4)[:,:,:3].sum())
                accum[j,k,3]+=float(diff[:,:32].reshape(-1,8,4)[:,:,3].sum())
        count+=len(ids)
    den=np.array([32,12,24,8],dtype=np.float64)
    result=np.sqrt(accum/np.maximum(count,1)/den[None,None,:])
    return dict(samples=count,horizons_seconds=[.2,.5,1.],
        variables=['hydro_all_dimensionless','motion_mixed_SI','hydro_force_dimensionless','hydro_yaw_torque_dimensionless'],
        rmse_model=result[:,0].tolist(),rmse_persistence_constant_velocity=result[:,1].tolist())

def train_world(d,norm,seed):
    out=RUN/'world'/str(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    torch.manual_seed(seed);model=World().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);sample=sampler(d,seed)
    n={k:torch.as_tensor(v,device=DEVICE) for k,v in norm.items()}
    val=np.flatnonzero(d['split']==1);test=np.flatnonzero(d['split']==2)
    best=float('inf');curve=[]
    for epoch in range(1,PROTOCOL['world_epochs']+1):
        check_stop();model.train();losses=[]
        for _ in range(PROTOCOL['updates_per_epoch']):
            loss=world_loss(model,d,next(sample),n);assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss))
        model.eval();v=score_world(model,d,val,n)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=v);curve.append(row)
        if v<best:
            best=v;torch.save(dict(model=model.state_dict(),norm=norm,seed=seed,epoch=epoch,dataset_sha256=dataset_sha()),out/'best.pt')
        write(out/'curve.json',curve);print(json.dumps(dict(world_seed=seed,**row)),flush=True)
    saved=torch.load(out/'best.pt',map_location=DEVICE,weights_only=False);model.load_state_dict(saved['model'])
    scores={mode:score_world(model,d,test,n,mode) for mode in ('normal','zero','shuffle')}
    diagnostics=world_diagnostics(model,d,test,n)
    write(out/'diagnostics.json',diagnostics)
    write(out/'complete.json',dict(passed=True,selected_epoch=saved['epoch'],test_loss=scores,
        n8_fit=False,normalization_train_only=True,encoder_hash=state_hash(model.encoder),
        labels='future 8x4 normalized fluid wrenches plus relative XYZ displacement, rotation vector, terminal linear velocity and angular velocity',
        horizons_steps=[10,25,50],true_flow_supervision=False))

@torch.no_grad()
def score_policy(model,d,ids):
    total=np.zeros(4)
    for start in range(0,len(ids),512):
        ix=ids[start:start+512];p=model(tensor(d,'observation',ix),tensor(d,'history',ix))
        total+=(p-tensor(d,'action',ix)).square().sum(0).cpu().numpy()
    return dict(samples=len(ids),action_mse=float(total.sum()/len(ids)/4),per_channel_mse=(total/len(ids)).tolist())

def train_bc(d,norm,seed,kind):
    out=RUN/'bc'/str(seed)/kind;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    saved=torch.load(RUN/'world'/str(seed)/'best.pt',map_location='cpu',weights_only=False)
    model=Policy(kind,norm,seed,saved['model']).to(DEVICE)
    frozen=state_hash(model.encoder) if kind=='C' else None
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    sample=sampler(d,seed);splits=[np.flatnonzero(d['split']==i) for i in range(3)]
    best=float('inf');curve=[]
    for epoch in range(1,PROTOCOL['bc_epochs']+1):
        check_stop();model.train();losses=[]
        for _ in range(PROTOCOL['updates_per_epoch']):
            ix=next(sample);p=model(tensor(d,'observation',ix),tensor(d,'history',ix))
            loss=(p-tensor(d,'action',ix)).square().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();losses.append(float(loss))
        if kind=='C':assert frozen==state_hash(model.encoder)
        model.eval();v=score_policy(model,d,splits[1]);row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=v);curve.append(row)
        if v['action_mse']<best:
            best=v['action_mse'];torch.save(dict(model=model.state_dict(),norm=norm,kind=kind,seed=seed,epoch=epoch,world_model_sha256=sha(RUN/'world'/str(seed)/'best.pt')),out/'best.pt')
        write(out/'curve.json',curve);print(json.dumps(dict(kind=kind,seed=seed,**row)),flush=True)
    s=torch.load(out/'best.pt',map_location=DEVICE,weights_only=False);model.load_state_dict(s['model'])
    if kind=='C':assert frozen==state_hash(model.encoder)
    scores={name:score_policy(model,d,ix) for name,ix in zip(('train','validation','test'),splits)}
    scores['by_n']={str(n):score_policy(model,d,splits[2][d['n'][splits[2]]==n]) for n in TRAIN_N}
    write(out/'complete.json',dict(passed=True,kind=kind,seed=seed,selected_epoch=s['epoch'],scores=scores,
        updates=PROTOCOL['bc_epochs']*PROTOCOL['updates_per_epoch'],samples_presented=PROTOCOL['bc_epochs']*PROTOCOL['updates_per_epoch']*PROTOCOL['batch_size'],
        encoder_frozen=kind=='C',frozen_hash=frozen,n8_fit=False))

def load_policy(seed,kind):
    saved=torch.load(RUN/'bc'/str(seed)/kind/'best.pt',map_location='cpu',weights_only=False)
    w=torch.load(RUN/'world'/str(seed)/'best.pt',map_location='cpu',weights_only=False)
    p=Policy(kind,saved['norm'],seed,w['model']).to(DEVICE);p.load_state_dict(saved['model']);p.eval();return p

def main(seed):
    assert seed in SEEDS
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    d=data();norm=normalization(d)
    train_world(d,norm,seed)
    for kind in ('A','B','C'):train_bc(d,norm,seed,kind)
    write(RUN/'learning_workers'/f'{seed}.json',dict(passed=True,seed=seed,dataset_sha256=dataset_sha(),n8_updates=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    main(p.parse_args().seed)
