"""Fresh success trajectories with actual simulator-oracle fluid wrenches."""
import json
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation
from .config import RUN, TRAIN_N, DEVICE, PROTOCOL, write, sha, check_stop
from .environment import base, distance3, reached
from .teacher import load_actor,setup_cells,prescribe,obs,validate
from .interaction import InteractionSensor, FEATURE_DIM

@torch.no_grad()
def snapshot(env,sensor):
    f=env.scene['fish'];wave=env.action_manager.get_term('wave')
    return {k:v.detach().cpu().numpy().copy() for k,v in dict(
        observation=base(env),feature=sensor.read(),legacy_feature=env.feature(),qpos=env.sim.data.qpos,
        qvel=env.sim.data.qvel,flow=env.flow,action_smoothed=wave.smoothed,
        xyz=f.data.root_link_pos_w,torque=f.data.actuator_force).items()}

def collect(n):
    assert n in TRAIN_N
    out=RUN/'data'/f'N{n}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    qualified=RUN/'frozen_teachers'/f'N{n}.pt'
    assert qualified.is_file(),'Missing immutable Teacher snapshot'
    counts=np.zeros((12,6),int);attempts=np.zeros_like(counts)
    # HDF5 is append-only across resumptions; already committed complete groups
    # are counted from metadata. Incomplete groups are never training inputs.
    rawfile=out/'episodes.h5'
    round_id=0;records=[]
    with h5py.File(rawfile,'a') as h:
        h.attrs['n']=n;h.attrs['teacher_sha256']=sha(qualified);h.attrs['schema']='pre_state_t -> executed_action_t -> next_state_t+1'
        for key,g in h.items():
            if g.attrs.get('committed',False):
                counts[int(g.attrs['morph']),int(g.attrs['flow_kind'])]+=1
                round_id=max(round_id,int(g.attrs['round'])+1)
        if (out/'attempts.json').exists():
            records=json.loads((out/'attempts.json').read_text());
            for r in records:
                attempts[r['morph'],r['flow_kind']]+=1
                round_id=max(round_id,int(r['round'])+1)
    env,actor=load_actor(n,qualified,12*6*2)
    sensor=InteractionSensor(env)
    projection=sensor.check_projection()
    assert projection['passed'],projection
    write(out/'fluid_projection_audit.json',projection)
    try:
        while round_id<PROTOCOL['collection_rounds']:
            check_stop()
            seed=PROTOCOL['collection_seed_base']+n*100000+round_id
            setup=setup_cells(env,seed,replicates=2)
            pending=np.ones(env.num_envs,dtype=bool)
            active=pending.copy();ends=np.zeros(env.num_envs,int);succeeded=np.zeros(env.num_envs,bool)
            blocks=[]
            with torch.inference_mode():
                for step in range(env.max_episode_length):
                    prescribe(env,step);pre=snapshot(env,sensor);raw=actor(obs(env))
                    _,reward,term,trunc,_=env.step(raw);post=snapshot(env,sensor)
                    # No reset occurs before both endpoints and applied control
                    # have been copied. A and C cannot access any post fields.
                    block={f'state_{k}':v for k,v in pre.items()}
                    block.update({f'next_{k}':v for k,v in post.items()})
                    block['action_raw']=raw.cpu().numpy().copy()
                    block['ctrl']=env.sim.data.ctrl.cpu().numpy().copy()
                    block['reward']=reward.cpu().numpy().copy()
                    block['terminated']=term.cpu().numpy().copy();block['truncated']=trunc.cpu().numpy().copy()
                    blocks.append(block)
                    done=(term|trunc).cpu().numpy()
                    for i in np.flatnonzero(done&active):
                        ends[i]=step+1;succeeded[i]=bool(reached(env)[i]);active[i]=False
                    if not active.any():break
                    if done.any():env.reset(env_ids=torch.tensor(np.flatnonzero(done),device=DEVICE))
                    if step%500==499:check_stop()
            arrays={k:np.stack([b[k] for b in blocks]) for k in blocks[0]}
            with h5py.File(rawfile,'a') as h:
                for lane in np.flatnonzero(pending):
                    m,k=int(setup['morph'][lane]),int(setup['flow_kind'][lane]);length=int(ends[lane])
                    assert length>0
                    a={key:value[:length,lane] for key,value in arrays.items()}
                    path=float(np.linalg.norm(np.diff(np.concatenate((a['state_xyz'][:1],a['next_xyz'])),axis=0),axis=-1).sum())
                    initial=float(np.linalg.norm(np.r_[setup['goals'][lane],.5]-a['state_xyz'][0]))
                    max_spin=float(np.linalg.norm(a['next_observation'][:,5:8],axis=-1).max())
                    max_speed=float(np.linalg.norm(a['next_qvel'][:,:3],axis=-1).max())
                    finite=all(np.isfinite(v).all() for v in a.values())
                    joint_max=float(np.abs(a['next_qpos'][:,7:]).max())
                    valid=bool(succeeded[lane] and finite and length>=PROTOCOL['collection_min_steps'] and max_spin<6. and max_speed<3. and joint_max<.82 and initial/max(path,1e-6)>.12)
                    record=dict(round=round_id,lane=int(lane),morph=m,flow_kind=k,seed=seed,length=length,
                        success=bool(succeeded[lane]),accepted=valid,path_length_m=path,path_efficiency=initial/max(path,1e-6),
                        max_spin_rad_s=max_spin,max_speed_m_s=max_speed,max_joint_rad=joint_max)
                    attempts[m,k]+=1;records.append(record)
                    if valid:
                        name=f'r{round_id:04d}_l{lane:04d}'
                        if name in h:raise RuntimeError('Refusing duplicate/partial episode name')
                        g=h.create_group(name)
                        for key,v in a.items():g.create_dataset(key,data=v,compression='lzf')
                        # Assign the complete scenario to one split before any
                        # windows are constructed, including WM pretraining.
                        split=int((round_id*2+int(lane)%2)%10)
                        split=1 if split==8 else 2 if split==9 else 0
                        for key,value in dict(**record,n=n,split=split,teacher_sha256=sha(qualified)).items():g.attrs[key]=value
                        g.attrs['goal_xyz']=np.r_[setup['goals'][lane],.5]
                        g.attrs['flow_params']=setup['flow_params'][lane]
                        g.attrs['committed']=True;counts[m,k]+=1
                h.flush()
            write(out/'attempts.json',records)
            write(out/'progress.json',dict(round=round_id,counts=counts.tolist(),attempts=attempts.tolist(),accepted=int(counts.sum()),min_cell=int(counts.min())))
            print(json.dumps(dict(n=n,round=round_id,accepted=int(counts.sum()),min_cell=int(counts.min()))),flush=True)
            round_id+=1
        write(out/'complete.json',dict(passed=bool(counts.sum()>0),scope='success_subset_only_not_full_coverage',n=n,accepted=int(counts.sum()),counts=counts.tolist(),attempts=attempts.tolist(),raw_sha256=sha(rawfile),teacher_sha256=sha(qualified),attempts_per_cell=PROTOCOL['collection_attempts_per_cell'],full_coverage_qualified=False))
    finally:env.close()

def rotation(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))

def build_dataset():
    assert all(json.loads((RUN/'data'/f'N{n}'/'complete.json').read_text())['passed'] for n in TRAIN_N)
    dest=RUN/'dataset.h5'
    if dest.exists():
        assert json.loads((RUN/'dataset_audit.json').read_text())['passed'];return
    keys=dict(history=(50,FEATURE_DIM),observation=(17,),action=(4,),future_actions=(50,4),
        current=(17,),labels=(3,44),flow_body=(3,),n=(),morph=(),flow_kind=(),split=(),episode=(),t=())
    tmp=dest.with_suffix('.building.h5');episode_id=0;rows=0;split_counts=np.zeros(3,int);morph_counts={}
    with h5py.File(tmp,'w') as out:
        for k,shape in keys.items():
            integer=k in ('n','morph','flow_kind','split','episode','t')
            out.create_dataset(k,shape=(0,)+shape,maxshape=(None,)+shape,dtype='i8' if integer else 'f4',chunks=(16,)+shape,compression='lzf')
        for n in TRAIN_N:
            with h5py.File(RUN/'data'/f'N{n}'/'episodes.h5','r') as h:
                for name,g in h.items():
                    if not g.attrs.get('committed',False):continue
                    assert n!=8 and g.attrs['accepted'] and g.attrs['success']
                    f=g['state_feature'][:];nf=g['next_feature'][:];o=g['state_observation'][:]
                    q=g['state_qpos'][:];nq=g['next_qpos'][:];raw=g['action_raw'][:];L=len(f)
                    # The prescribed flow may advance at the next control-step
                    # boundary; loads can change at fixed qpos. Applied wave
                    # controls and mechanical state must remain continuous.
                    assert np.allclose(f[1:,-4:],nf[:-1,-4:],atol=2e-5),f'Broken wave alignment {n}/{name}'
                    assert np.allclose(q[1:],nq[:-1],atol=2e-5)
                    assert f.shape[-1]==FEATURE_DIM and o.shape[-1]==17
                    idx=np.arange(49,L-50,10)
                    if len(idx)==0:continue
                    R=rotation(q[:,3:7]);nR=rotation(nq[:,3:7]);batch={k:[] for k in keys}
                    for t in idx:
                        future=np.array([t+9,t+24,t+49])
                        # XYZ displacement, exact body-frame relative rotation
                        # vector, and actual endpoint v/omega.
                        rel=np.einsum('ij,hjk->hik',R[t].T,nR[future])
                        rotvec=Rotation.from_matrix(rel).as_rotvec()
                        disp=(nq[future,:3]-q[t,:3])@R[t]
                        velocity=np.einsum('ij,hj->hi',R[t].T,g['next_qvel'][future,:3])
                        # angular velocity from feature is actual body-frame omega;
                        # label's frame is future body, documented explicitly.
                        # The first 32 labels are future fluid loads; movement
                        # is an auxiliary consequence of the action-conditioned flow.
                        label=np.concatenate((nf[future,:32],disp,rotvec,velocity,
                                              g['next_observation'][future,5:8]),-1)
                        values=dict(history=f[t-49:t+1],observation=o[t],action=np.tanh(raw[t]),
                            future_actions=nf[t:t+50,-4:],current=o[t],labels=label,
                            flow_body=g['state_flow'][t]@R[t],n=n,morph=int(g.attrs['morph']),
                            flow_kind=int(g.attrs['flow_kind']),split=int(g.attrs['split']),episode=episode_id,t=int(t))
                        for key,value in values.items():batch[key].append(value)
                    for key in keys:
                        out[key].resize(rows+len(idx),axis=0);out[key][rows:]=np.asarray(batch[key])
                    rows+=len(idx);split_counts[int(g.attrs['split'])]+=len(idx)
                    morph_counts[str(n)]=morph_counts.get(str(n),0)+1;episode_id+=1
        out.attrs['policy_input_whitelist']='observation,history'
        out.attrs['timing']='history ends at pre-state t; action_t executes afterwards; future fields are WM head inputs/labels only'
        out.flush()
    assert (split_counts>0).all() and set(morph_counts)==set(map(str,TRAIN_N))
    tmp.replace(dest)
    with h5py.File(dest,'r') as h:
        splits=[set(h['episode'][:][h['split'][:]==i]) for i in range(3)]
        assert all(not (splits[i]&splits[j]) for i in range(3) for j in range(i+1,3))
        assert set(np.unique(h['n'][:]))==set(TRAIN_N)
    write(RUN/'dataset_audit.json',dict(passed=True,scope='success_subset',full_coverage_qualified=False,samples=rows,episodes=episode_id,episodes_by_n=morph_counts,
        split_samples=split_counts.tolist(),n8_in_fit=False,episode_splits_disjoint=True,
        base_contract='17D same as prior experiment',history_contract='50x36 eight conservatively rebinned simulator fluid wrenches and prior executed smoothed physical wave',
        teacher_privileged_in_policy_input=False,future_in_policy_input=False,
        wm_fit_and_normalization_train_only=True,wm_selection_validation_only=True,sha256=sha(dest)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,choices=TRAIN_N);p.add_argument('--build',action='store_true')
    a=p.parse_args();build_dataset() if a.build else collect(a.n)
