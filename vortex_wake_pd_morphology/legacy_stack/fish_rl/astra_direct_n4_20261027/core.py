from __future__ import annotations
import sys, json, math, hashlib, csv
from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('sac_navigation_reference',ROOT/'1027_formal/environment.py')
sac=importlib.util.module_from_spec(spec);spec.loader.exec_module(sac)
CFG=sac.CFG
FLOWS=['still','constant','rotating','strength','reversing','dynamic3d']

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
                              default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def servo_xml():
    root=ET.parse(HERE/'source_snapshot/fish_N4_entity.xml').getroot()
    root.insert(1,ET.parse(ROOT/'1027_formal/source_snapshot/fish_N4.xml').getroot().find('option'))
    return ET.tostring(root,encoding='unicode')

class ServoEnv(sac.PathFishEnv):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.model=mujoco.MjModel.from_xml_string(servo_xml())
        self.data=mujoco.MjData(self.model);self.head=self.model.body('head').id
    def step(self,target_angles):
        q=np.asarray(target_angles,dtype=float)
        if q.shape!=(4,) or not np.isfinite(q).all():raise ValueError('Four finite target angles required')
        q=np.clip(q,-.75,.75)
        # Reuse exact SAC physical stepping, flow timing, observation, episode
        # conditions. Its normalized action mapping multiplies by 12, so this
        # produces ctrl=q radians on the ORIGINAL position servo, not torque.
        return super().step(q/CFG['torque_limit_Nm'])

def wrap(a):return (a+np.pi)%(2*np.pi)-np.pi

class Controller:
    def __init__(self,params):
        self.p=params;self.phase=0.;self.yaw_rate=0.;self.filtered_yaw=None;self.last_bias=0.
        self.index=np.arange(4)
    def action(self,e):
        p=self.p;pos=e.data.xpos[e.head];R=e.data.xmat[e.head].reshape(3,3)
        yaw=math.atan2(R[1,0],R[0,0]);delta=e.target-pos[:2];distance=np.linalg.norm(delta)
        if self.filtered_yaw is None:self.filtered_yaw=yaw
        self.filtered_yaw=wrap(self.filtered_yaw+p.get('yaw_filter',.1)*wrap(yaw-self.filtered_yaw))
        self.yaw_rate=.9*self.yaw_rate+.1*float(e.data.qvel[5])
        # Ground-velocity objective, compensated by physical water velocity.
        wind=e.model.opt.wind[:2]
        speed=p.get('speed',.25)
        desired=speed*delta/max(distance,1e-8)-p.get('flow_gain',0.)*wind
        heading=math.atan2(desired[1],desired[0]);error=wrap(heading-self.filtered_yaw)
        bias=np.clip(p.get('kp',.3)*error-p.get('kd',.15)*self.yaw_rate,-p.get('bias_limit',.3),p.get('bias_limit',.3))
        bias*=p.get('sign',1.)
        self.last_bias=.85*self.last_bias+.15*bias
        slowdown=np.clip(distance/p.get('slow_radius',.8),p.get('min_frequency_ratio',.4),1.)
        self.phase+=2*np.pi*p.get('frequency',1.5)*slowdown*e.dt
        amplitude=p.get('amplitude',.5)
        profile=p.get('head_ratio',.5)+(1-p.get('head_ratio',.5))*self.index/3
        q=self.last_bias+amplitude*profile*np.sin(self.phase-self.index*p.get('lag',1.0))
        ramp=min((e.age+1)*e.dt/.4,1.)
        return np.clip(q*ramp,-.75,.75)

def scenario(seed,flow,fixed=False):
    e=ServoEnv(stage='fixed_still' if fixed else 'random_still',flow=flow)
    _,info=e.reset(seed=seed);sc=info['scenario'];e.close();return sc

def rollout(params,sc,trace=False,seconds=60,method='direct',model=None):
    e=(sac.PathFishEnv if method=='sac' else ServoEnv)(stage='random_still',flow=sc['flow'],seconds=seconds)
    obs,_=e.reset(seed=0,options=sc)
    controller=Controller(params)
    min_xy=float(obs[0]);min_xyz=float(np.hypot(min_xy,e.data.qpos[2]-.5))
    first_xy=first_xyz=first_both=None
    energy=0.;spin_sq=0.;max_spin=0.;max_speed=0.;saturation=0.
    records=[]
    def record(command):
        return np.concatenate(([e.age*e.dt],e.data.qpos.copy(),e.data.qvel.copy(),command,
            e.data.actuator_force.copy(),e.model.opt.wind.copy(),[float(e.observation()[0]),
            np.hypot(float(e.observation()[0]),e.data.qpos[2]-.5),e.dwell,energy]))
    if trace:records.append(record(np.zeros(4)))
    while True:
        a=model.predict(obs,deterministic=True)[0] if method=='sac' else (np.zeros(4) if method=='passive' else controller.action(e))
        obs,_,term,trunc,info=e.step(a)
        # Mechanical work estimate sampled at each 20 ms control boundary;
        # stored torque and joint velocity allow independent recomputation.
        energy+=float(np.sum(np.abs(e.data.actuator_force*e.data.qvel[6:])))*e.dt
        spin=np.linalg.norm(e.data.qvel[3:6]);speed=np.linalg.norm(e.data.qvel[:3])
        spin_sq+=spin**2;max_spin=max(max_spin,spin);max_speed=max(max_speed,speed)
        saturation+=float(np.mean(np.abs(e.data.ctrl)>=.749)) if method!='sac' else float(np.mean(np.abs(a)>=.999))
        xy,xyz=info['distance'],info['distance_xyz'];min_xy=min(min_xy,xy);min_xyz=min(min_xyz,xyz)
        if xy<CFG['success_radius_xy_m'] and first_xy is None:first_xy=e.age*e.dt
        if xyz<CFG['success_radius_xyz_m'] and first_xyz is None:first_xyz=e.age*e.dt
        if xy<CFG['success_radius_xy_m'] and xyz<CFG['success_radius_xyz_m'] and first_both is None:first_both=e.age*e.dt
        if trace:records.append(record(e.data.ctrl.copy()))
        if term or trunc:break
    result=dict(**info['result'],method=method,energy_abs_mechanical_J=energy,
        angular_velocity_rms_rad_s=float(np.sqrt(spin_sq/e.age)),angular_velocity_max_rad_s=float(max_spin),
        max_linear_speed_m_s=float(max_speed),command_saturation_fraction=saturation/e.age,
        min_distance_xy_m=min_xy,min_distance_xyz_m=min_xyz,
        distance_only_xy_reached=first_xy is not None,distance_only_xyz_reached=first_xyz is not None,
        distance_only_both_reached=first_both is not None,first_distance_xy_s=first_xy,
        first_distance_xyz_s=first_xyz,first_distance_both_s=first_both)
    if result['success']:reason='strict_success_10_consecutive_steps'
    elif result['physical_failure']:
        if abs(e.data.qpos[2]-.5)>2.5:reason='physical_depth_boundary'
        elif np.linalg.norm(e.data.qpos[:2])>14:reason='physical_xy_boundary'
        elif speed>5:reason='physical_speed_limit'
        else:reason='physics_reset_or_other_failure'
    elif first_both is not None:reason='passed_distance_threshold_but_dwell_or_velocity_failed'
    elif first_xy is not None:reason='xy_near_target_but_depth_failed'
    else:reason='timeout_without_reaching_xy_threshold'
    result['reason']=reason
    e.close()
    return result,np.asarray(records) if trace else None

TRACE_COLUMNS=(['time_s']+['x','y','z','qw','qx','qy','qz']+[f'joint_{i}_actual_rad' for i in range(4)]
    +['vx','vy','vz','wx','wy','wz']+[f'joint_{i}_velocity_rad_s' for i in range(4)]
    +[f'joint_{i}_command' for i in range(4)]+[f'joint_{i}_torque_Nm' for i in range(4)]
    +['flow_x','flow_y','flow_z','distance_xy','distance_xyz','success_dwell_steps','energy_abs_mechanical_J'])

def save_trace(out,key,trace,sc,result):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert trace.shape[1]==len(TRACE_COLUMNS)
    np.savez_compressed(out/f'{key}.npz',data=trace,columns=np.array(TRACE_COLUMNS),
                        scenario=json.dumps(sc),result=json.dumps(result))
    np.savetxt(out/f'{key}.csv',trace,delimiter=',',header=','.join(TRACE_COLUMNS),comments='',fmt='%.9g')

def summary(rows):
    wins=[r for r in rows if r['success']]
    return dict(episodes=len(rows),successes=len(wins),success_rate=len(wins)/len(rows),
        distance_only_xy_success_rate=float(np.mean([r['distance_only_xy_reached'] for r in rows])),
        distance_only_xyz_success_rate=float(np.mean([r['distance_only_xyz_reached'] for r in rows])),
        distance_only_both_success_rate=float(np.mean([r['distance_only_both_reached'] for r in rows])),
        mean_time_to_target_s=float(np.mean([r['seconds'] for r in wins])) if wins else None,
        mean_final_distance_xy_m=float(np.mean([r['final_distance_m'] for r in rows])),
        mean_final_distance_xyz_m=float(np.mean([r['final_distance_xyz_m'] for r in rows])),
        mean_energy_abs_mechanical_J=float(np.mean([r['energy_abs_mechanical_J'] for r in rows])),
        mean_angular_velocity_rms_rad_s=float(np.mean([r['angular_velocity_rms_rad_s'] for r in rows])),
        physical_failure_rate=float(np.mean([r['physical_failure'] for r in rows])),
        reasons={k:sum(r['reason']==k for r in rows) for k in sorted({r['reason'] for r in rows})})
