"""Independent N=4 torque task; native MuJoCo, unchanged legacy water/flow."""
from __future__ import annotations
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
FLOWS = CFG['flows']

def model_xml():
    # Read only the experiment snapshot; never rewrite original XML/checkpoints.
    root = ET.parse(HERE / 'source_snapshot/fish_N4_entity.xml').getroot()
    original = ET.parse(HERE / 'source_snapshot/fish_N4.xml').getroot()
    root.insert(1, ET.fromstring(ET.tostring(original.find('option'))))
    actuator = root.find('actuator')
    actuator.clear()
    for i in range(4):
        ET.SubElement(actuator, 'motor', name=f'joint_{i}', joint=f'joint_{i}',
                      gear='1', ctrllimited='true', ctrlrange='-12 12',
                      forcelimited='true', forcerange='-12 12')
    return ET.tostring(root, encoding='unicode')

def flow_velocity(kind, base, phase, freq, seconds):
    """Exact numpy port of fresh_abc_20261022/environment.py:Bucket.flow.

    Same parameter ranges (.02-.18 m/s, .15-.40 Hz); heldout reset uses
    frequency*2.5, as in original base_env.py. No new flow physics.
    """
    p = 2 * math.pi * freq * seconds + phase
    angle = .9 * np.sin(p)
    c, s = np.cos(angle), np.sin(angle)
    rotated = np.array([base[0]*c-base[1]*s, base[0]*s+base[1]*c, base[2]])
    if kind == 'still': return np.zeros(3)
    if kind == 'constant': return base.copy()
    if kind == 'rotating': return rotated
    if kind == 'strength': return base * (1 + .65*np.sin(p))
    if kind == 'reversing': return base * (1 if seconds < 15 else -1)
    if kind == 'dynamic3d':
        v = rotated * (1 + .5*np.sin(p)); v[2] = .025*np.cos(p*1.3)
        return v
    if kind == 'heldout':
        v = rotated * (1 + .65*np.sin(p*2.71)); v[2] = .04*np.sin(p*1.91)
        return v
    raise ValueError(kind)

def geometry(position_xy, yaw, start, target):
    line = target - start
    u = np.clip(np.dot(position_xy-start, line) / np.dot(line, line), 0., 1.)
    closest = start + u*line
    delta = target-position_xy
    c, s = np.cos(yaw), np.sin(yaw)
    world_to_body = np.array([[c, s], [-s, c]])
    local_target = world_to_body @ delta
    local_path = world_to_body @ (closest-position_xy)
    return np.array([np.linalg.norm(delta), math.atan2(local_target[1], local_target[0]),
                     *local_path, np.linalg.norm(closest-position_xy)], dtype=np.float32)

class PathFishEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, stage='fixed_still', flow=None, seconds=None):
        self.cfg = CFG
        self.model = mujoco.MjModel.from_xml_string(model_xml())
        self.data = mujoco.MjData(self.model)
        self.head = self.model.body('head').id
        self.dt = CFG['control_dt']
        self.substeps = round(self.dt/self.model.opt.timestep)
        assert abs(self.substeps*self.model.opt.timestep-self.dt) < 1e-10
        self.max_steps = round((seconds or CFG['episode_seconds'])/self.dt)
        self.stage, self.flow_override = stage, flow
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([0,-np.pi,-np.inf,-np.inf,0], np.float32),
                                           np.array([np.inf,np.pi,np.inf,np.inf,np.inf], np.float32))
        self.ended = True

    def set_stage(self, stage):
        # Applied to task/flow only at the next episode reset.
        self.stage = stage

    def observation(self):
        rotation = self.data.xmat[self.head].reshape(3,3)
        yaw = math.atan2(rotation[1,0], rotation[0,0])
        return geometry(self.data.xpos[self.head,:2], yaw, self.start, self.target)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        options = options or {}
        fixed = self.stage == 'fixed_still'
        self.start = np.array(CFG['fixed_start'] if fixed else r.uniform(-.25,.25,2), dtype=float)
        if fixed:
            self.target = np.array(CFG['fixed_target'], dtype=float); yaw = 0.
        else:
            yaw = r.uniform(-np.pi,np.pi)
            angle = yaw + r.uniform(-np.pi,np.pi)
            radius = r.uniform(1.5,6.)
            self.target = self.start + radius*np.array([np.cos(angle),np.sin(angle)])
        self.start = np.array(options.get('start', self.start), dtype=float)
        self.target = np.array(options.get('target', self.target), dtype=float)
        yaw = float(options.get('yaw', yaw))
        assert np.linalg.norm(self.target-self.start) > .5
        self.data.qpos[:3] = [*self.start, .5]
        self.data.qpos[3:7] = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
        # Fixed task is identical on every reset. Random task follows existing
        # joint/velocity ranges; no added roll/pitch curriculum or morphology.
        if not fixed:
            self.data.qpos[7:] = r.uniform(-.08,.08,4)
            self.data.qvel[:] = r.uniform(-.03,.03,self.model.nv)
        if 'qpos' in options: self.data.qpos[:] = options['qpos']
        if 'qvel' in options: self.data.qvel[:] = options['qvel']
        choices = {'fixed_still':['still'], 'random_still':['still'],
                   'constant':['still','constant'],
                   'varying':['still','constant','rotating','strength','reversing'],
                   'complex':FLOWS[:-1]}[self.stage]
        self.flow = options.get('flow', self.flow_override or str(r.choice(choices)))
        angle, speed = r.uniform(-np.pi,np.pi), r.uniform(.02,.18)
        self.base = np.array([speed*np.cos(angle),speed*np.sin(angle),0.])
        self.phase, self.freq = r.uniform(0,2*np.pi), r.uniform(.15,.40)
        if self.flow == 'heldout': self.freq *= 2.5
        self.base = np.array(options.get('flow_base',self.base))
        self.phase, self.freq = options.get('flow_phase',self.phase), options.get('flow_freq',self.freq)
        self.model.opt.wind[:] = flow_velocity(self.flow,self.base,self.phase,self.freq,0)
        mujoco.mj_forward(self.model,self.data)
        self.age = self.dwell = 0
        self.path_sum = self.path_max = self.depth_max = self.return_sum = 0.
        self.ended = False
        self.scenario = dict(start=self.start.tolist(),target=self.target.tolist(),yaw=yaw,
                             qpos=self.data.qpos.tolist(),qvel=self.data.qvel.tolist(),
                             flow=self.flow,flow_base=self.base.tolist(),flow_phase=self.phase,flow_freq=self.freq)
        return self.observation(), {'scenario':self.scenario}

    def step(self, action):
        if self.ended: raise RuntimeError('Episode ended: call reset before step')
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (4,) or not np.isfinite(a).all(): raise ValueError('Invalid torque action')
        a = np.clip(a,-1,1)
        before = float(self.observation()[0])
        self.model.opt.wind[:] = flow_velocity(self.flow,self.base,self.phase,self.freq,self.age*self.dt)
        self.data.ctrl[:] = CFG['torque_limit_Nm'] * a
        time_before = self.data.time
        mujoco.mj_step(self.model,self.data,nstep=self.substeps)
        # mj_step leaves position caches at the last substep's pre-integration
        # pose. Refresh them before building the next control-step observation.
        mujoco.mj_forward(self.model,self.data)
        self.age += 1
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        if not finite: raise FloatingPointError('Nonfinite MuJoCo state')
        obs = self.observation(); distance, deviation = float(obs[0]), float(obs[4])
        xyz = self.data.xpos[self.head]
        depth = abs(float(xyz[2])-.5)
        distance_xyz = float(np.hypot(distance,depth))
        speed = np.linalg.norm(self.data.qvel[:3]); spin = np.linalg.norm(self.data.qvel[3:6])
        sim_reset = abs(self.data.time-time_before-self.dt) > 1e-8
        failure = bool(sim_reset or np.linalg.norm(xyz[:2]) > 14 or depth > 2.5 or speed > 5)
        valid = distance < CFG['success_radius_xy_m'] and distance_xyz < CFG['success_radius_xyz_m'] and speed < 2 and spin < 4
        self.dwell = self.dwell+1 if valid else 0
        success = self.dwell >= CFG['success_dwell_steps'] and not failure
        terminated = bool(success or failure)
        truncated = bool(self.age >= self.max_steps and not terminated)
        w = CFG['reward']
        parts = dict(goal_progress=w['progress']*(before-distance),
                     path_following=-w['path_per_second']*self.dt*deviation**2,
                     action_regularization=-w['action_per_second']*self.dt*float(np.mean(a*a)),
                     time=-w['time_per_second']*self.dt,
                     terminal=w['success']*success+w['failure']*failure)
        reward = sum(parts.values())
        self.path_sum += deviation; self.path_max = max(self.path_max,deviation)
        self.depth_max = max(self.depth_max,depth); self.return_sum += reward
        self.ended = terminated or truncated
        info = dict(is_success=bool(success), physical_failure=failure, flow=self.flow,
                    distance=distance,distance_xyz=distance_xyz,path_deviation=deviation,
                    depth_deviation=depth,reward_parts=parts)
        if self.ended:
            info['result'] = dict(success=bool(success),physical_failure=failure,truncated=truncated,
                flow=self.flow,steps=self.age,seconds=self.age*self.dt,
                time_to_target_s=self.age*self.dt if success else None,
                mean_path_deviation_m=self.path_sum/self.age,max_path_deviation_m=self.path_max,
                final_distance_m=distance,final_distance_xyz_m=distance_xyz,
                max_depth_deviation_m=self.depth_max,episode_return=self.return_sum)
        return obs,float(reward),terminated,truncated,info
