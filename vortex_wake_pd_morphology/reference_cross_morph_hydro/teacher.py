"""Privileged PPO, independent validation, no student or N8 selection."""
from dataclasses import asdict
from pathlib import Path
import argparse, hashlib, json, time,signal
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from fish_rl.rl_cfg import make_rl_cfg
from .config import RUN, TRAIN_N, DEVICE, PROTOCOL, write, sha, status, check_stop
from .environment import DiverseFishEnv, distance3, reached
from .privilege import observation_dim

def obs(env):return TensorDict(env.observation_manager.compute(),batch_size=[env.num_envs])

def runner_for(env,out):
    cfg=make_rl_cfg();cfg.seed=20261020+env.n
    cfg.algorithm.learning_rate=PROTOCOL['teacher_learning_rate'];cfg.algorithm.entropy_coef=.002
    cfg.algorithm.gamma=PROTOCOL['teacher_gamma'];cfg.algorithm.schedule='fixed'
    cfg.actor.distribution_cfg['init_std']=.25
    cfg.actor.distribution_cfg['std_range']=(.08,.5)
    wrapper=RslRlVecEnvWrapper(env,clip_actions=100.)
    return MjlabOnPolicyRunner(wrapper,asdict(cfg),log_dir=str(out/'tensorboard'),device=DEVICE)

@torch.no_grad()
def project_exploration(runner,initial=False,repair_only=False):
    for name,p in runner.alg.actor.named_parameters():
        if name.endswith('distribution.std_param'):
            if repair_only:
                bad=(p<=.08)|(p>=.5)
                p[bad]=p[bad].clamp(.10,.35)
            else:p.clamp_(.10 if initial else .081,.35 if initial else .499)

def initialize(runner,n,out):
    source=RUN/'parent_checkpoints'/f'N{n}.pt'
    s=torch.load(source,map_location=DEVICE,weights_only=False)
    for name,model in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
        src=s[name+'_state_dict'];dst=model.state_dict()
        for key,v in dst.items():
            previous=src[key]
            if v.shape==previous.shape:v.copy_(previous)
            elif key=='mlp.0.weight':
                v.zero_();v[:,:previous.shape[1]].copy_(previous)
            elif key.startswith('obs_normalizer.') and v.ndim==2:
                v.fill_(0. if key.endswith('_mean') else 1.)
                v[:,:previous.shape[1]].copy_(previous)
            else:raise ValueError(f'Unsupported warm start {key}: {v.shape} {previous.shape}')
        model.load_state_dict(dst,strict=True)
        model.obs_normalizer.update=lambda x:None
    # New observation columns start at zero, preserving the deterministic actor.
    # Reset the value output for the corrected reward/gamma; reuse its features.
    with torch.no_grad():
        runner.alg.critic.mlp[-1].weight.zero_();runner.alg.critic.mlp[-1].bias.zero_()
    project_exploration(runner,initial=True)
    write(out/'initialization.json',dict(source=str(source),sha256=sha(source),
        source_iteration=int(s['iter']),old_dim=30+2*n,new_dim=observation_dim(n),
        additional='physical morphology, inertia, actuation limits/utilization, XYZ target, relative local flow and time derivatives',
        normalizer='old prefix copied/frozen; appended training-only calibration',new_first_layer_columns_zero=True,
        optimizer='fresh Adam: observation shape and reward objective changed',critic='hidden features reused, value output reset'))

@torch.no_grad()
def calibrate(runner,env,out,steps=180):
    old=30+2*env.n;s=s2=None;count=0
    env.level=1.;env.target_radius_max=6.;env.reset()
    # Only training RNG and N3/4/5/6, never validation/acceptance/N8 samples.
    runner.alg.actor.eval()
    for _ in range(steps):
        o=obs(env);x=o['actor'][:,old:].double()
        a=x.sum(0);b=x.square().sum(0);s=a if s is None else s+a;s2=b if s2 is None else s2+b;count+=len(x)
        _,_,term,trunc,_=env.step(runner.alg.actor(o));done=term|trunc
        if done.any():env.reset(env_ids=torch.nonzero(done).flatten())
    mean=s/count;std=(s2/count-mean.square()).clamp_min(0).sqrt().clamp_min(.1)
    for model in (runner.alg.actor,runner.alg.critic):
        model.obs_normalizer._mean[:,old:]=mean.float()
        model.obs_normalizer._std[:,old:]=std.float()
        model.obs_normalizer._var[:,old:]=std.float().square()
    write(out/'normalization_audit.json',dict(training_only=True,n8=False,validation=False,samples=count,
        old_prefix_frozen=True,static_physics_names=env.physical_names,extra_mean=mean.cpu().tolist(),extra_std=std.cpu().tolist(),
        spatial_flow='uniform physical wind; per-body relative flow includes actual local body velocity; no fabricated shear'))

def save(runner,path,it):
    runner.current_learning_iteration=it
    s=runner.alg.save();s.update(iter=it,infos=dict(experiment='size_flow_v2',n8_training=False,
        observation_version=PROTOCOL['teacher_observation_version'],training_state=getattr(runner,'training_state',{})))
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(s,tmp);tmp.replace(path)

def load_actor(n,path,count=1):
    env=DiverseFishEnv(n,count,random_morph=False)
    runner=runner_for(env,RUN/'load_tmp'/f'N{n}')
    s=torch.load(path,map_location=DEVICE,weights_only=False)
    runner.alg.actor.load_state_dict(s['actor_state_dict']);runner.alg.actor.eval()
    return env,runner.alg.actor

def setup_cells(env,seed,*,replicates=12,suite='ID',difficulty=1.,target_radius_max=6.):
    """Paired initial states and independent prescribed per-lane flow formulas."""
    n_morph=len(env.meta);count=env.num_envs
    assert count==n_morph*6*replicates
    rng=np.random.default_rng(seed)
    ids=torch.arange(count,device=DEVICE)
    morph=np.repeat(np.arange(n_morph),6*replicates)
    kind=np.tile(np.repeat(np.arange(6),replicates),n_morph)
    env.random_morph=False;env.fixed_schedule=True
    env.set_morph(ids,torch.tensor(morph,device=DEVICE))
    env.reset()
    q=env.sim.data.qpos;yaw=rng.uniform(-np.pi,np.pi,count)
    bearing=rng.uniform(-np.pi,np.pi,count);radius=rng.uniform(1.5,6,count)
    q[:,:3]=torch.tensor([0,0,.5],device=DEVICE);q[:,3:7]=0
    q[:,3]=torch.tensor(np.cos(yaw/2),device=DEVICE);q[:,6]=torch.tensor(np.sin(yaw/2),device=DEVICE)
    env.sim.data.qvel[:]=0
    radius=1.5+(radius-1.5)*(target_radius_max-1.5)/4.5
    goals=radius[:,None]*np.stack((np.cos(yaw+bearing),np.sin(yaw+bearing)),-1)
    goal=env.command_manager.get_term('goal');goal.goal_xy[:]=torch.tensor(goals,device=DEVICE)
    env.sim.data.mocap_pos[:,goal.marker_id,:2]=goal.goal_xy
    angle=rng.integers(0,4,count)*np.pi/2
    if suite=='direction':angle+=np.pi/4
    mag=rng.uniform(.03,.45,count)
    if suite=='strength':mag=rng.uniform(.48,.65,count)
    v=np.stack((mag*np.cos(angle),mag*np.sin(angle),np.zeros(count)),-1)
    v[kind==2,:2]=0
    v[:,2]=np.where((kind==2)|(kind==3),rng.uniform(.005,.025,count)*rng.choice([-1,1],count),0.)
    if suite=='direction':
        # Global yaw is a symmetry of this task. Merely rotating horizontal
        # world axes is insufficient evidence of new dynamics. Include steep
        # constant oblique flow at a seen total speed but unseen inclination.
        oblique=kind==3;speed=rng.uniform(.20,.40,int(oblique.sum()))
        v[oblique,:2]=speed[:,None]/np.sqrt(2)*np.stack((np.cos(angle[oblique]),np.sin(angle[oblique])),-1)
        v[oblique,2]=speed/np.sqrt(2)*rng.choice([-1,1],int(oblique.sum()))
    v[kind==0]=0
    amp=np.zeros_like(v);dynamic=kind>=4
    amp[dynamic,:2]=v[dynamic,:2]*.4;v[dynamic,:2]*=.6
    amp[:,2]=np.where(kind==5,rng.uniform(.02,.10,count),0.)
    if suite=='strength':
        v[:,2]=np.where((kind==2)|(kind==3),rng.uniform(.035,.060,count)*rng.choice([-1,1],count),0.)
        amp[:,2]=np.where(kind==5,rng.uniform(.12,.16,count),0.)
    periods=rng.uniform(2,6,count)
    if suite in ('dynamic3d','size_dynamic3d'):
        # All conditions become continuously changing 3D flow at unseen azimuth,
        # greater speed, shorter periods and larger vertical oscillation.
        angle+=np.pi/4;mag=rng.uniform(.20,.60,count)
        v=np.stack((.6*mag*np.cos(angle),.6*mag*np.sin(angle),np.zeros(count)),-1)
        amp=np.stack((.4*mag*np.cos(angle),.4*mag*np.sin(angle),rng.uniform(.10,.16,count)),-1)
        periods=rng.uniform(.6,1.5,count)
    phases=rng.uniform(0,2*np.pi,count)
    if difficulty<1.:
        v[:,:2]*=(.03+.42*difficulty)/.45;amp[:,:2]*=(.03+.42*difficulty)/.45
        v[:,2]*=(.003+.022*difficulty)/.025;amp[:,2]*=(.01+.09*difficulty)/.1
    env.flow_kind[:]=torch.tensor(kind,device=DEVICE)
    env.flow_start[:]=torch.tensor(v,device=DEVICE);env.flow_amp[:]=torch.tensor(amp,device=DEVICE)
    env.flow_period[:]=torch.tensor(periods,device=DEVICE);env.flow_phase[:]=torch.tensor(phases,device=DEVICE)
    env.flow_age[:]=0
    env.flow[:]=env.flow_start+env.flow_amp*env.flow_phase[:,None].sin()
    env.sim.forward();env.prev_d.copy_(distance3(env));env.dwell[:]=0
    initial=q.cpu().numpy().copy()
    flow_params=np.concatenate((v,amp,periods[:,None],phases[:,None]),-1).astype(np.float32)
    h=lambda a:hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
    return dict(morph=morph,flow_kind=kind,initial=initial,goals=goals.astype(np.float32),
        flow_params=flow_params,initial_hash=h(np.concatenate((initial,goals),-1)),flow_hash=h(flow_params),seed=seed,suite=suite)

def prescribe(env,step):
    env.flow_age.fill_(step*.02)
    phase=env.flow_phase+2*torch.pi*(step*.02)/env.flow_period
    env.flow.copy_(env.flow_start+env.flow_amp*phase[:,None].sin())

@torch.no_grad()
def validate(actor,env,seed,replicates=12,out=None,*,difficulty=1.,target_radius_max=6.):
    setup=setup_cells(env,seed,replicates=replicates,difficulty=difficulty,target_radius_max=target_radius_max)
    count=env.num_envs;active=np.ones(count,bool);rows=[None]*count
    max_spin=np.zeros(count);max_speed=np.zeros(count);path=np.zeros(count)
    min_dist=np.full(count,np.inf);min_xy=np.full(count,np.inf)
    max_tracking=np.zeros(count);max_util=np.zeros(count);max_joint=np.zeros(count)
    previous=env.scene['fish'].data.root_link_pos_w.cpu().numpy().copy()
    traces=[];qtraces=[];actor.eval()
    for step in range(env.max_episode_length):
        prescribe(env,step);raw=actor(obs(env))
        _,_,term,trunc,_=env.step(raw)
        f=env.scene['fish'];xyz=f.data.root_link_pos_w.cpu().numpy().copy()
        path[active]+=np.linalg.norm(xyz-previous,axis=1)[active];previous=xyz
        max_spin[active]=np.maximum(max_spin[active],f.data.root_link_ang_vel_b.norm(dim=-1).cpu().numpy()[active])
        max_speed[active]=np.maximum(max_speed[active],f.data.root_link_lin_vel_w.norm(dim=-1).cpu().numpy()[active])
        d=distance3(env).cpu().numpy();min_dist[active]=np.minimum(min_dist[active],d[active])
        xy=np.linalg.norm(xyz[:,:2]-setup['goals'],axis=-1);min_xy[active]=np.minimum(min_xy[active],xy[active])
        w=env.action_manager.get_term('wave');limits=env.sim.model.actuator_forcerange[:,:,1].abs().clamp_min(1e-3)
        tracking=((w.q_target-f.data.joint_pos).abs()/w._range).amax(-1).cpu().numpy()
        util=(f.data.actuator_force.abs()/limits).amax(-1).cpu().numpy()
        joint=f.data.joint_pos.abs().amax(-1).cpu().numpy()
        max_tracking[active]=np.maximum(max_tracking[active],tracking[active])
        max_util[active]=np.maximum(max_util[active],util[active]);max_joint[active]=np.maximum(max_joint[active],joint[active])
        if step%5==0:
            qp=env.sim.data.qpos.cpu().numpy().copy();qp[~active]=np.nan;qtraces.append(qp)
            trace=xyz.copy();trace[~active]=np.nan;traces.append(trace)
        done=(term|trunc).cpu().numpy();success=reached(env).cpu().numpy()
        d=distance3(env).cpu().numpy()
        for i in np.flatnonzero(done&active):
            rows[i]=dict(lane=int(i),morph=int(setup['morph'][i]),flow_kind=int(setup['flow_kind'][i]),
                success=bool(success[i]),seconds=(step+1)*.02,distance3_m=float(d[i]),depth_error_m=float(abs(xyz[i,2]-.5)),
                path_length_m=float(path[i]),max_spin_rad_s=float(max_spin[i]),max_speed_m_s=float(max_speed[i]),
                length_m=float(env.meta[int(setup['morph'][i]),0]),initial_goal_distance_m=float(np.linalg.norm(setup['goals'][i])),
                min_distance3_m=float(min_dist[i]),min_distance_xy_m=float(min_xy[i]),
                max_tracking_error_fraction=float(max_tracking[i]),max_torque_utilization=float(max_util[i]),max_joint_rad=float(max_joint[i]),
                physically_unreachable_proven=False,
                reason='reach' if success[i] else 'timeout' if bool(trunc[i]) else 'physical_failure')
            active[i]=False
        if not active.any():break
        if done.any():env.reset(env_ids=torch.tensor(np.flatnonzero(done),device=DEVICE))
        if step%500==499:check_stop()
    assert all(r is not None for r in rows)
    cells=[]
    for m in range(len(env.meta)):
        for k in range(6):
            rr=[r for r in rows if r['morph']==m and r['flow_kind']==k]
            cells.append(dict(morph=m,flow_kind=k,count=len(rr),success_rate=float(np.mean([r['success'] for r in rr]))))
    rate=float(np.mean([r['success'] for r in rows]));worst=min(r['success_rate'] for r in cells)
    full=difficulty==1. and target_radius_max==6. and replicates>=PROTOCOL['teacher_gate_trials_per_cell']
    result=dict(n=env.n,trials=len(rows),success_rate=rate,worst_cell_success_rate=worst,
        passed=full and rate>=PROTOCOL['teacher_gate_overall'] and worst>=PROTOCOL['teacher_gate_per_cell'],
        full_difficulty=full,difficulty=difficulty,target_radius_max=target_radius_max,
        cells=cells,initial_hash=setup['initial_hash'],flow_hash=setup['flow_hash'],seed=seed,
        no_censoring=True,success_is_xyz_and_dwell=True)
    if out:
        write(out/'summary.json',result);write(out/'episodes.json',rows)
        np.savez_compressed(out/'rollout.npz',xyz=np.asarray(traces),qpos=np.asarray(qtraces),frame_dt=np.float64(.1),goals=setup['goals'],initial=setup['initial'],flow_params=setup['flow_params'])
        from .diagnostics import stratify
        write(out/'stratified.json',stratify(rows,setup['flow_params']))
    return result

def train(n,iterations=None,envs=None,smoke=False,baseline_only=False):
    raise RuntimeError('Frozen-Teacher collection only: no Teacher RL authorized in this run')
    assert n in TRAIN_N
    torch.set_num_threads(2)
    iterations=iterations or PROTOCOL['teacher_iterations'];count=envs or PROTOCOL['teacher_envs']
    out=RUN/('smoke_teacher' if smoke else 'teachers')/f'N{n}';out.mkdir(parents=True,exist_ok=True)
    env=DiverseFishEnv(n,count,seed=20261020+n);runner=runner_for(env,out)
    start_it=0;state=dict(stage=0,stage_since=0,full_since=None,history=[])
    if (out/'latest.pt').exists():
        s=torch.load(out/'latest.pt',map_location=DEVICE,weights_only=False)
        assert s['infos']['observation_version']==PROTOCOL['teacher_observation_version']
        runner.alg.load(s,None,strict=True);start_it=int(s['iter'])
        project_exploration(runner,repair_only=True)
        state=s['infos']['training_state']
        for model in (runner.alg.actor,runner.alg.critic):model.obs_normalizer.update=lambda x:None
    else:
        initialize(runner,n,out)
        if not smoke:calibrate(runner,env,out)
        runner.training_state=state;save(runner,out/'initial.pt',0)
    runner.training_state=state
    if baseline_only:
        val_env=DiverseFishEnv(n,864,seed=710000+n,random_morph=False)
        try:validate(runner.alg.actor,val_env,710000+n,out=out/'baseline_v1_policy')
        finally:val_env.close();env.close()
        return
    env.level=PROTOCOL['curriculum_levels'][state['stage']]
    env.target_radius_max=(3.,4.,5.,6.)[state['stage']];env.reset()
    o=obs(env);recent=[];returns=torch.zeros(count,device=DEVICE);started=time.time()
    best=max([r['score'] for r in state['history']],default=-1.);passed=False;reason='maximum_safety_budget'
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    it=start_it
    try:
        if not smoke and not (out/'baseline_v1_policy'/'summary.json').exists():
            val_env=DiverseFishEnv(n,864,seed=710000+n,random_morph=False)
            try:validate(runner.alg.actor,val_env,710000+n,out=out/'baseline_v1_policy')
            finally:val_env.close()
            o=obs(env)
        for it in range(start_it+1,iterations+1):
            check_stop()
            warmup=not smoke and it<=PROTOCOL['teacher_critic_warmup_updates']
            runner.alg.actor.requires_grad_(not warmup)
            runner.alg.train_mode()
            with torch.inference_mode():
                for _ in range(24):
                    a=runner.alg.act(o)
                    _,r,term,trunc,extras=env.step(a);done=term|trunc
                    terminal=obs(env)
                    # 60 s is a real task failure horizon (age is observed).
                    # Do not bootstrap a hypothetical continuation after it.
                    corrected=r
                    extras.pop('time_outs',None)
                    runner.alg.process_env_step(terminal,corrected,done,extras)
                    returns+=r
                    if done.any():
                        ids=torch.nonzero(done).flatten()
                        for i in ids.tolist():recent.append(dict(success=bool(reached(env)[i]),return_=float(returns[i]),seconds=float(env.episode_length_buf[i])*.02))
                        recent=recent[-1000:];returns[ids]=0;env.reset(env_ids=ids)
                    o=obs(env)
                runner.alg.compute_returns(o)
            loss=runner.alg.update()
            project_exploration(runner)
            assert all(np.isfinite(float(v)) for v in loss.values()),'Nonfinite PPO loss'
            assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()),'Nonfinite actor'
            row=dict(n=n,iteration=it,curriculum=env.level,sim_steps=it*24*count,
                curriculum_stage=state['stage'],target_radius_max=env.target_radius_max,
                critic_only_warmup=warmup,
                recent_success=float(np.mean([r['success'] for r in recent])) if recent else None,
                elapsed_s=time.time()-started,loss={k:float(v) for k,v in loss.items()})
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if it%10==0 or smoke:write(out/'progress.json',row);print(json.dumps(row),flush=True)
            if it%50==0 or it==iterations:save(runner,out/'latest.pt',it)
            if not smoke and (it%PROTOCOL['teacher_eval_every']==0 or it==iterations):
                save(runner,out/f'iteration_{it:05d}.pt',it)
                # Validation always uses full difficulty and separate fresh seeds.
                val_env=DiverseFishEnv(n,12*6*PROTOCOL['teacher_gate_trials_per_cell'],seed=710000+n,random_morph=False)
                try:val=validate(runner.alg.actor,val_env,710000+n,out=out/'validation'/f'iter_{it:05d}')
                finally:val_env.close()
                score=val['success_rate']+.25*val['worst_cell_success_rate']
                state['history'].append(dict(iteration=it,score=score,success=val['success_rate'],worst=val['worst_cell_success_rate'],stage=state['stage']))
                if score>best:best=score;save(runner,out/'best.pt',it)
                write(out/'gate_latest.json',dict(iteration=it,**val));print(json.dumps(dict(validation=val)),flush=True)
                if val['passed']:
                    save(runner,out/'qualified.pt',it);write(out/'qualified.json',dict(iteration=it,**val));passed=True;break
                if state['stage']<3:
                    probe_env=DiverseFishEnv(n,216,seed=730000+n,random_morph=False)
                    try:probe=validate(runner.alg.actor,probe_env,730000+n,replicates=3,
                        out=out/'curriculum_probes'/f'iter_{it:05d}',difficulty=env.level,target_radius_max=env.target_radius_max)
                    finally:probe_env.close()
                    stage=state['stage'];age=it-state['stage_since']
                    mastered=probe['success_rate']>=PROTOCOL['curriculum_probe_success'][stage] and probe['worst_cell_success_rate']>=PROTOCOL['curriculum_probe_worst'][stage]
                    if mastered and age>=PROTOCOL['curriculum_min_updates']:
                        state['stage']+=1;state['stage_since']=it
                        if state['stage']==3:state['full_since']=it
                        env.level=PROTOCOL['curriculum_levels'][state['stage']]
                        env.target_radius_max=(3.,4.,5.,6.)[state['stage']]
                        env.reset();returns[:]=0
                    elif age>=PROTOCOL['curriculum_max_stage_updates']:
                        reason='curriculum_mastery_stalled';break
                else:
                    full=[r['score'] for r in state['history'] if r['stage']==3]
                    span=PROTOCOL['teacher_plateau_evaluations']
                    if len(full)>=span*2 and it-state['full_since']>=PROTOCOL['teacher_min_full_updates']:
                        improvement=max(full[-span:])-max(full[-2*span:-span])
                        if improvement<PROTOCOL['teacher_plateau_min_gain']:
                            reason='full_difficulty_validation_plateau';break
                write(out/'training_state.json',state);save(runner,out/'latest.pt',it)
                o=obs(env)
        save(runner,out/'latest.pt',it)
        write(out/'complete.json',dict(completed=True,qualified=passed,iteration=it,smoke=smoke,
            reason='quality_gate_passed' if passed else 'smoke_only' if smoke else reason,
            stopping_uses_validation=True,max_budget_is_safety_cap=True))
    finally:
        if it>start_it:save(runner,out/'latest.pt',it)
        env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True,choices=TRAIN_N)
    p.add_argument('--iterations',type=int);p.add_argument('--envs',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--baseline-only',action='store_true')
    a=p.parse_args();train(a.n,a.iterations,a.envs,a.smoke,a.baseline_only)
