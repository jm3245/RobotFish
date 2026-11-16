"""Paired frozen-policy rollout. N8 is instantiated only after fitting ends."""
import json,hashlib
import numpy as np
import torch
from .config import RUN,SEEDS,DEVICE,PROTOCOL,write,sha,check_stop
from .environment import DiverseFishEnv,base,distance3,reached
from .teacher import setup_cells,prescribe
from .learn import load_policy,state_hash
from .interaction import InteractionSensor,FEATURE_DIM
from fish_rl.shared_model.student_rollout import fixed_warmup_raw

def paired_setup(env,seed,replicates,suite):
    # Build one deterministic bank then duplicate each scenario across A/B/C.
    s=setup_cells(env,seed,replicates=replicates*3,suite=suite)
    cells=len(env.meta)*6
    sources=np.concatenate([np.arange(c*replicates*3,c*replicates*3+replicates) for c in range(cells)])
    indices=np.tile(sources,3);ids=torch.arange(env.num_envs,device=DEVICE)
    ix=torch.tensor(indices,device=DEVICE)
    for key in ('morph','flow_kind','initial','goals','flow_params'):s[key]=s[key][indices].copy()
    env.set_morph(ids,torch.tensor(s['morph'],device=DEVICE));env.sim.data.qpos[:]=torch.tensor(s['initial'],device=DEVICE)
    env.sim.data.qvel[:]=0
    goal=env.command_manager.get_term('goal');goal.goal_xy[:]=torch.tensor(s['goals'],device=DEVICE)
    env.sim.data.mocap_pos[:,goal.marker_id,:2]=goal.goal_xy
    for key in ('flow_start','flow_amp','flow_period','flow_phase','flow_kind'):
        getattr(env,key).copy_(getattr(env,key)[ix].clone())
    prescribe(env,0);env.sim.forward();env.prev_d.copy_(distance3(env));env.dwell[:]=0
    width=env.num_envs//3
    initial=np.concatenate((s['initial'][:width],s['goals'][:width]),-1)
    s['initial_hash']=hashlib.sha256(initial.tobytes()).hexdigest()
    s['flow_hash']=hashlib.sha256(s['flow_params'][:width].tobytes()).hexdigest()
    for j in (1,2):
        sl=slice(j*width,(j+1)*width)
        assert np.array_equal(s['initial'][:width],s['initial'][sl])
        assert np.array_equal(s['flow_params'][:width],s['flow_params'][sl])
    return s

@torch.inference_mode()
def evaluate_one(seed,suite,n=8,replicates=None):
    trials=replicates or PROTOCOL['eval_trials_per_cell']
    outputs=[RUN/'evaluation'/f'N{n}'/str(seed)/suite/k for k in ('A','B','C')]
    if all((p/'summary.json').exists() for p in outputs):return
    ood=suite in ('size','size_dynamic3d');morphs=4 if ood else 12
    width=morphs*6*trials;count=width*3
    models=[load_policy(seed,k) for k in ('A','B','C')];before=[state_hash(p) for p in models]
    env=DiverseFishEnv(n,count,seed=900000+n,evaluation=True,ood=ood,random_morph=False)
    sensor=InteractionSensor(env)
    projection=sensor.check_projection()
    assert projection['passed'],projection
    s=paired_setup(env,seed+PROTOCOL['evaluation_seed_offset']+n,trials,suite)
    history=torch.zeros((count,50,FEATURE_DIM),device=DEVICE)
    active=np.ones(count,bool);rows=[None]*count
    origin=env.scene['fish'].data.root_link_pos_w.cpu().numpy().copy();previous=origin.copy()
    path=np.zeros(count);cross2=np.zeros(count);steps=np.zeros(count);max_spin=np.zeros(count);energy=np.zeros(count)
    direction=np.c_[s['goals'],np.full(count,.5)]-origin
    initial_distance=np.linalg.norm(direction,axis=-1);direction/=initial_distance[:,None]
    traces=[]
    try:
        for step in range(env.max_episode_length):
            prescribe(env,step)
            if step<50:raw=fixed_warmup_raw().to(DEVICE).repeat(count,1)
            else:
                o=base(env);raw=torch.cat([torch.atanh(model(o[j*width:(j+1)*width],history[j*width:(j+1)*width]).clamp(-.999999,.999999)) for j,model in enumerate(models)])
            _,_,term,trunc,_=env.step(raw)
            history=torch.roll(history,-1,1);history[:,-1]=sensor.read()
            f=env.scene['fish'];xyz=f.data.root_link_pos_w.cpu().numpy().copy()
            path[active]+=np.linalg.norm(xyz-previous,axis=-1)[active];previous=xyz
            disp=xyz-origin;cross=np.linalg.norm(disp-np.sum(disp*direction,axis=-1)[:,None]*direction,axis=-1)
            cross2[active]+=cross[active]**2;steps[active]+=1
            spin=f.data.root_link_ang_vel_b.norm(dim=-1).cpu().numpy();max_spin[active]=np.maximum(max_spin[active],spin[active])
            power=(f.data.actuator_force*f.data.joint_vel).abs().sum(-1).cpu().numpy();energy[active]+=power[active]*.02
            if step%10==0:
                tr=xyz.copy();tr[~active]=np.nan;traces.append(tr)
            done=(term|trunc).cpu().numpy();reach=reached(env).cpu().numpy();dist=distance3(env).cpu().numpy()
            for i in np.flatnonzero(done&active):
                rows[i]=dict(id=int(i%width),morph=int(s['morph'][i]),flow_kind=int(s['flow_kind'][i]),
                    success=bool(reach[i]),arrival_time_s=(step+1)*.02 if reach[i] else None,
                    capped_time_s=(step+1)*.02 if reach[i] else PROTOCOL['episode_seconds'],
                    duration_s=(step+1)*.02,final_distance3_m=float(dist[i]),
                    path_length_m=float(path[i]),path_deviation_rms_m=float(np.sqrt(cross2[i]/steps[i])),
                    depth_error_m=float(abs(xyz[i,2]-.5)),path_efficiency=float(initial_distance[i]/max(path[i],1e-6)),
                    max_spin_rad_s=float(max_spin[i]),absolute_joint_work_J=float(energy[i]),
                    reason='reach' if reach[i] else 'timeout' if bool(trunc[i]) else 'physical_failure')
                active[i]=False
            if not active.any():break
            if done.any():
                ids=torch.tensor(np.flatnonzero(done),device=DEVICE);env.reset(env_ids=ids);history[ids]=0
            if step%500==499:
                check_stop();print(json.dumps(dict(stage='eval',n=n,seed=seed,suite=suite,step=step+1,active=int(active.sum()))),flush=True)
        assert all(r is not None for r in rows),'Censored evaluations cannot silently become failures'
        assert before==[state_hash(p) for p in models]
        trace=np.asarray(traces)
        for j,(kind,out) in enumerate(zip(('A','B','C'),outputs)):
            rr=rows[j*width:(j+1)*width];ok=[r for r in rr if r['success']]
            result=dict(n=n,seed=seed,suite=suite,kind=kind,count=width,successes=len(ok),success_rate=len(ok)/width,
                mean_arrival_time_success_s=float(np.mean([r['arrival_time_s'] for r in ok])) if ok else None,
                mean_capped_time_s=float(np.mean([r['capped_time_s'] for r in rr])),
                mean_path_deviation_rms_m=float(np.mean([r['path_deviation_rms_m'] for r in rr])),
                mean_path_length_m=float(np.mean([r['path_length_m'] for r in rr])),
                initial_hash=s['initial_hash'],flow_hash=s['flow_hash'],parameters_unchanged=True,n8_training_updates=0,
                success_full_xyz=True,shared_warmup_seconds=1.,timeout_seconds=PROTOCOL['episode_seconds'])
            write(out/'summary.json',result);write(out/'episodes.json',rr)
            np.savez_compressed(out/'rollout.npz',xyz=trace[:,j*width:(j+1)*width],goals=s['goals'][:width],
                initial=s['initial'][:width],flow_params=s['flow_params'][:width])
    finally:env.close()

def main(seed):
    assert seed in SEEDS and json.loads((RUN/'learning_complete.json').read_text())['passed']
    torch.set_num_threads(2)
    paths=list((RUN/'bc').glob('*/*/best.pt'))+list((RUN/'world').glob('*/best.pt'))
    before=json.loads((RUN/'evaluation_frozen_manifest.json').read_text())
    assert before=={str(p):sha(p) for p in paths}
    for suite in PROTOCOL['suites']:evaluate_one(seed,suite)
    assert before=={str(p):sha(p) for p in paths}
    write(RUN/'evaluation_workers'/f'{seed}.json',dict(passed=True,parameters_unchanged=True,n8_updates=0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    main(p.parse_args().seed)
