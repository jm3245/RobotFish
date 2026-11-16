"""8D -> 1D SAC steering; first head entry into the XY target ends the episode."""
from pathlib import Path
import importlib.util
import json
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE.parent / 'fish_rl' / 'astra_direct_n4_20261027'
spec = importlib.util.spec_from_file_location('astra_original', BASELINE / 'core.py')
astra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(astra)
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
PARAMS = json.loads((BASELINE / 'frozen/controllers.json').read_text())
FLOWS = CFG['training_flows']


class SteeringWave(astra.Controller):
    """Original waveform equations with an externally supplied physical bias.

    b is AFTER the old PD sign conversion, BEFORE the original bias filter.
    Phase, approach slowdown, profile, ramp and clamps match Controller.action.
    """
    def action_with_bias(self, e, bias):
        p = self.p
        distance = np.linalg.norm(e.target - e.data.xpos[e.head, :2])
        self.last_bias = .85 * self.last_bias + .15 * float(np.clip(bias, -.3, .3))
        slowdown = np.clip(distance / p.get('slow_radius', .8), p.get('min_frequency_ratio', .4), 1.)
        self.phase += 2 * np.pi * p.get('frequency', 1.5) * slowdown * e.dt
        amplitude = p.get('amplitude', .5)
        profile = p.get('head_ratio', .5) + (1-p.get('head_ratio', .5)) * self.index / 3
        q = self.last_bias + amplitude * profile * np.sin(self.phase-self.index*p.get('lag', 1.0))
        ramp = min((e.age+1)*e.dt/.4, 1.)
        return np.clip(q*ramp, -.75, .75)


class SteeringEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, flow=None):
        super().__init__()
        self.physics = astra.ServoEnv(stage='complex', flow=flow, seconds=60)
        self.action_space = spaces.Box(-.3, .3, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)
        self.wave = None
        self.previous_raw_bias = 0.
        self.reward_sum = 0.

    def observation(self):
        e = self.physics
        R = e.data.xmat[e.head].reshape(3, 3)
        return np.asarray([*e.data.xpos[e.head, :2], math.atan2(R[1, 0], R[0, 0]),
                           e.data.qvel[5], *e.target, *e.model.opt.wind[:2]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _, info = self.physics.reset(seed=seed, options=options)
        self.wave = SteeringWave(PARAMS[self.physics.flow])
        self.previous_raw_bias = 0.
        self.reward_sum = 0.
        return self.observation(), info

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (1,) or not np.isfinite(a).all():
            raise ValueError('Expected one finite steering bias in radians')
        bias = float(np.clip(a[0], -.3, .3))
        e = self.physics
        before = np.linalg.norm(e.target-e.data.xpos[e.head, :2])
        targets = self.wave.action_with_bias(e, bias)
        _, _, _, trunc, info = e.step(targets)
        success = bool(info['distance'] < CFG['success_xy_radius_m']
                       and not info['physical_failure'])
        term = bool(success or info['physical_failure'])
        if success:
            e.ended = True
        info['is_success'] = success
        w = CFG['reward']
        parts = dict(progress=w['progress']*(before-info['distance']),
                     success=w['success']*success,
                     failure=w['failure']*info['physical_failure'],
                     time=-w['time_per_second']*e.dt,
                     bias_smoothness=-w['raw_bias_delta_squared']*((bias-self.previous_raw_bias)/.3)**2)
        reward = float(sum(parts.values()))
        self.reward_sum += reward
        self.previous_raw_bias = bias
        info.update(steering_bias_raw_rad=bias, steering_bias_filtered_rad=self.wave.last_bias,
                    steering_reward_parts=parts)
        if term or trunc:
            info['result'] = dict(success=success, physical_failure=info['physical_failure'],
                                  truncated=trunc, flow=e.flow, steps=e.age, seconds=e.age*e.dt,
                                  time_to_target_s=e.age*e.dt if success else None,
                                  mean_path_deviation_m=e.path_sum/e.age,
                                  max_path_deviation_m=e.path_max,
                                  final_distance_m=info['distance'],
                                  final_distance_xyz_m=info['distance_xyz'],
                                  max_depth_deviation_m=e.depth_max,
                                  original_env_return=e.return_sum,
                                  episode_return=self.reward_sum)
        return self.observation(), reward, term, trunc, info

    def close(self):
        self.physics.close()
