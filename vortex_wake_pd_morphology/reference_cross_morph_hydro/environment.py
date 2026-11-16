from __future__ import annotations
import math
import numpy as np
import torch
import warp as wp
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.utils.lab_api.math import quat_apply_inverse
from fish_rl.omni_env_cfg import make_omni_env_cfg
from fish_rl.omni_multi_wave_action import MultiWaveActionCfg
from fish_rl import mdp
from fish_rl.shared_model.student_rollout import motion_feature
from .config import PROTOCOL, TRAIN_N, DEVICE
from .morphology import entity_xml, parameters, FIELDS
from .privilege import append,static_physics,observation_dim

def delta(env):
    p=env.scene['fish'].data.root_link_pos_w
    return torch.cat((env.command_manager.get_term('goal').goal_xy-p[:,:2],.5-p[:,2:3]),-1)

def distance3(env):return torch.linalg.vector_norm(delta(env),dim=-1)

def base(env):
    f=env.scene['fish'];w=env.action_manager.get_term('wave')
    # Keep the CURRENT 17D contract exactly. No added depth/flow/morphology.
    return torch.cat((mdp.target_body(env),f.data.root_link_lin_vel_b,f.data.root_link_ang_vel_b,
        mdp.obs_down_reference(env),w.smoothed,w.phase.sin()[:,None],w.phase.cos()[:,None]),-1)

def legacy_privileged(env):
    old=torch.cat((mdp.obs_target(env),mdp.obs_linear_velocity(env),mdp.obs_angular_velocity(env),
        mdp.obs_down_reference(env),mdp.obs_joint_pos(env),mdp.obs_joint_vel(env),mdp.obs_last_action(env),mdp.obs_phase(env)),-1)
    if not hasattr(env,'flow'):
        return torch.cat((old,torch.zeros((env.num_envs,3+3+6),device=env.device)),-1)
    q=env.scene['fish'].data.root_link_quat_w
    flow=quat_apply_inverse(q,env.flow)/.45
    # old horizontal-flow channels kept first for legacy warm start.
    speed=torch.linalg.vector_norm(env.flow[:,:2],dim=-1);a=torch.atan2(env.flow[:,1],env.flow[:,0])
    yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    oldflow=torch.stack((speed/.30,(a-yaw).sin(),(a-yaw).cos()),-1)
    oldflow=torch.where((speed>1e-7)[:,None],oldflow,0.)
    return torch.cat((old,oldflow,flow,env.morph,delta(env)[:,2:3]),-1)

def privileged(env):return append(env,legacy_privileged(env))

def reached(env):
    return env.dwell>=PROTOCOL['arrival_dwell_steps'] if hasattr(env,'dwell') else torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)

def progress(env):
    d=distance3(env)
    if not hasattr(env,'prev_d'):return torch.zeros_like(d)
    r=(env.prev_d-d)/env.step_dt;env.prev_d.copy_(d);return r

def reach_reward(env):return reached(env).float()
def time_cost(env):return torch.ones(env.num_envs,device=env.device)
def spin(env):return env.scene['fish'].data.root_link_ang_vel_b.square().sum(-1)
def effort(env):
    f=env.scene['fish']
    limit=env.sim.model.actuator_forcerange[:,:,1].abs().clamp_min(1e-3)
    return (f.data.actuator_force/limit).square().mean(-1)

def depth_cost(env):return delta(env)[:,2].abs()
def physical_failure(env):
    f=env.scene['fish'];p=f.data.root_link_pos_w
    return mdp.diverged(env,speed_limit=5.) | (p[:,2]-.5).abs().gt(2.5) | mdp.out_of_bounds(env,radius=14.)

def configuration(n,count,index=0,ood=False):
    cfg=make_omni_env_cfg(num_envs=count)
    path=entity_xml(n,index,ood)
    cfg.scene.entities['fish']=EntityCfg(spec_fn=lambda:mujoco.MjSpec.from_file(str(path)),
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,.5)),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=tuple(f'joint_{i}' for i in range(n))),)))
    cfg.actions['wave']=MultiWaveActionCfg(entity_name='fish',n_joints=n)
    cfg.auto_reset=False;cfg.is_finite_horizon=True;cfg.episode_length_s=PROTOCOL['episode_seconds']
    for group in cfg.observations.values():
        group.terms={'privileged':ObservationTermCfg(func=privileged)};group.enable_corruption=False
    # RewardManager multiplies rates by dt. Progress was already a step delta
    # in v1, accidentally reducing dense progress by another factor of 50.
    cfg.rewards=dict(progress=RewardTermCfg(func=progress,weight=4.),
        time=RewardTermCfg(func=time_cost,weight=-.12),
        depth=RewardTermCfg(func=depth_cost,weight=-.30),
        energy=RewardTermCfg(func=effort,weight=-.015),
        spin=RewardTermCfg(func=spin,weight=-.008),
        smooth=RewardTermCfg(func=mdp.smoothness,weight=-.03),
        limit=RewardTermCfg(func=mdp.near_limit,weight=-.05))
    cfg.terminations['reach']=TerminationTermCfg(func=reached)
    cfg.terminations['diverged']=TerminationTermCfg(func=physical_failure)
    cfg.metrics=dict(distance=MetricsTermCfg(func=distance3,reduce='last'))
    return cfg

class DiverseFishEnv(ManagerBasedRlEnv):
    def __init__(self,n,count,*,seed=20261020,evaluation=False,ood=False,random_morph=True):
        if n not in TRAIN_N and not (evaluation and n==8):raise ValueError('N8 must never enter training')
        self.n=n
        cfg=configuration(n,count,ood=ood);cfg.seed=seed
        super().__init__(cfg,device=DEVICE)
        self.n=n;self.ood=ood;self.random_morph=random_morph;self.level=1.
        self.fixed_schedule=False;self.rng=np.random.default_rng(seed)
        self.morph_ids=torch.zeros(count,dtype=torch.long,device=DEVICE)
        self.morph=torch.zeros((count,5),device=DEVICE)
        self.torque_scale=torch.ones((count,1),device=DEVICE)
        self.dwell=torch.zeros(count,dtype=torch.long,device=DEVICE)
        self.prev_d=distance3(self).clone()
        self.sim.wp_model.opt.wind=wp.zeros(count,dtype=wp.vec3,device=self.sim.wp_device)
        self.flow=wp.to_torch(self.sim.wp_model.opt.wind)
        self.sim.create_graph()
        self.flow_kind=torch.zeros(count,dtype=torch.long,device=DEVICE)
        self.flow_start=torch.zeros_like(self.flow);self.flow_end=torch.zeros_like(self.flow)
        self.flow_age=torch.zeros(count,device=DEVICE);self.flow_period=torch.ones(count,device=DEVICE)
        self.flow_amp=torch.zeros_like(self.flow)
        self.flow_phase=torch.zeros(count,device=DEVICE)
        self.site_ids,_=self.scene['fish'].find_sites([f's{i}' for i in range(8)],preserve_order=True)
        # CPU compilation computes exact mass/principal inertia from ellipsoids.
        models=[]
        for i in range(len(parameters(n,ood))):
            sc=Scene(configuration(n,count,i,ood).scene,device='cpu');models.append(sc.compile())
        self.sim.expand_model_fields(FIELDS)
        self.bank={}
        for field in FIELDS:
            if hasattr(self.sim.wp_model,field):
                dst=getattr(self.sim.model,field)
                vals=np.stack([getattr(m,field) for m in models])
                if field=='geom_aabb':vals=vals.reshape(len(models),-1,2,3)
                if tuple(dst.shape[1:])!=tuple(vals.shape[1:]):
                    if np.all(vals==vals[0]):continue
                    raise ValueError(f'Unexpected field shape {field}: {dst.shape}, {vals.shape}')
                self.bank[field]=torch.as_tensor(vals,device=DEVICE,dtype=dst.dtype)
        self.meta=torch.tensor([[r['length'],r['width_scale'],r['height_scale'],r['tail_scale'],r['proportion']] for _,r in parameters(n,ood)],device=DEVICE)
        self.scales=torch.tensor([[c.force_limit/12] for c,_ in parameters(n,ood)],device=DEVICE)
        wave=self.action_manager.get_term('wave')
        physical=[static_physics(m,c,meta,wave) for m,(c,meta) in zip(models,parameters(n,ood))]
        self.physical_bank=torch.tensor(np.stack([r[0] for r in physical]),device=DEVICE)
        self.physical_names=physical[0][1]
        self.observation_dim=observation_dim(n)
        self.target_radius_max=6.
        self._native_step=self.sim.step;self.sim.step=self._physics_step
        self.reset()

    def set_morph(self,ids,values):
        self.morph_ids[ids]=values;self.morph[ids]=self.meta[values]
        self.torque_scale[ids]=self.scales[values]
        for field,bank in self.bank.items():getattr(self.sim.model,field)[ids]=bank[values]
        self.sim.recompute_constants(RecomputeLevel.set_const)

    def sample_flow(self,ids):
        k=len(ids)
        if not k:return
        # Per-lane categories: still, constant horizontal, constant vertical,
        # constant shallow oblique, smoothly varying horizontal, dynamic 3D.
        kind=self.flow_kind[ids]
        direction=torch.randint(0,4,(k,),device=DEVICE).float()*math.pi/2
        mag=torch.rand(k,device=DEVICE)*(.03+.42*self.level)
        v=torch.stack((mag*direction.cos(),mag*direction.sin(),torch.zeros(k,device=DEVICE)),-1)
        vertical=(torch.rand(k,device=DEVICE)*2-1)*(.003+.022*self.level)
        v[kind==2,:2]=0;v[:,2]=torch.where((kind==2)|(kind==3),vertical,0.)
        v[kind==0]=0
        self.flow_start[ids]=v;self.flow_end[ids]=v
        self.flow_amp[ids]=0
        dynamic=kind>=4
        self.flow_start[ids,:2]=torch.where(dynamic[:,None],v[:,:2]*.6,v[:,:2])
        self.flow_amp[ids,:2]=torch.where(dynamic[:,None],v[:,:2]*.4,0.)
        self.flow_amp[ids,2]=torch.where(kind==5,(.01+.09*self.level)*torch.rand(k,device=DEVICE),0.)
        self.flow_age[ids]=0;self.flow_period[ids]=2+4*torch.rand(k,device=DEVICE)
        self.flow_phase[ids]=torch.rand(k,device=DEVICE)*2*math.pi
        self.flow[ids]=self.flow_start[ids]+self.flow_amp[ids]*self.flow_phase[ids,None].sin()

    def _physics_step(self):
        if not self.fixed_schedule:
            self.flow_age+=self.physics_dt
            phase=2*math.pi*self.flow_age/self.flow_period+self.flow_phase
            self.flow.copy_(self.flow_start+self.flow_amp*phase[:,None].sin())
        self._native_step()

    def reset(self,env_ids=None,**kwargs):
        result=super().reset(env_ids=env_ids,**kwargs)
        if hasattr(self,'bank'):
            ids=torch.arange(self.num_envs,device=DEVICE) if env_ids is None else env_ids
            if self.random_morph:
                values=torch.as_tensor(self.rng.integers(0,len(self.meta),len(ids)),device=DEVICE)
                self.set_morph(ids,values)
            self.dwell[ids]=0;self.prev_d[ids]=distance3(self)[ids]
            if not self.fixed_schedule:
                self.flow_kind[ids]=torch.as_tensor(self.rng.integers(0,6,len(ids)),device=DEVICE)
                self.sample_flow(ids)
                if self.target_radius_max<6.:
                    goal=self.command_manager.get_term('goal')
                    angle=torch.rand(len(ids),device=DEVICE)*2*math.pi
                    radius=1.5+torch.rand(len(ids),device=DEVICE)*(self.target_radius_max-1.5)
                    goal.goal_xy[ids]=radius[:,None]*torch.stack((angle.cos(),angle.sin()),-1)
                    self.sim.data.mocap_pos[ids,goal.marker_id,:2]=goal.goal_xy[ids]
            self.sim.forward();result=(self.observation_manager.compute(),result[1])
            self.prev_d[ids]=distance3(self)[ids]
        return result

    def step(self,action):
        result=super().step(action)
        f=self.scene['fish']
        valid=(distance3(self)<PROTOCOL['arrival_radius_m']) & (f.data.root_link_ang_vel_b.norm(dim=-1)<4.) & (f.data.root_link_lin_vel_w.norm(dim=-1)<2.)
        # Keep the original normal buffer: replacing it inside PPO's inference
        # context made later curriculum reset outside that context illegal.
        self.dwell.copy_(torch.where(valid,self.dwell+1,0))
        # The manager computed before this endpoint update. Evaluate the actual
        # endpoint now; no next-episode state is ever exposed to the recorder.
        reach=reached(self)
        term=result[2]|reach
        if reach.any():
            self.termination_manager._term_dones['reach'][:]=reach
            self.termination_manager._terminated_buf|=reach
            self.reset_terminated|=reach
            result[1][reach]+=30.
        failure=term&~reach
        result[1][failure]-=8.
        result[1][result[3]&~term]-=2.
        return result[0],result[1],term,result[3],result[4]

    def feature(self):return motion_feature(self,self.site_ids)
