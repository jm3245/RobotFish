"""The unchanged four-parameter wave action generalized to N joints."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from mjlab.managers.action_manager import ActionTerm

from fish_rl.wave_action import WaveAction, WaveActionCfg


@dataclass(kw_only=True)
class MultiWaveActionCfg(WaveActionCfg):
    n_joints: int

    def build(self, env):
        return MultiWaveAction(self, env)


class MultiWaveAction(WaveAction):
    cfg: MultiWaveActionCfg

    def __init__(self, cfg: MultiWaveActionCfg, env):
        ActionTerm.__init__(self, cfg, env)
        if not 0 <= cfg.keep_previous < 1 or cfg.n_joints < 2:
            raise ValueError("Invalid smoothing coefficient or joint count")
        names = [f"joint_{i}" for i in range(cfg.n_joints)]
        local_ids, found = self._entity.find_joints(names, preserve_order=True)
        if found != names:
            raise ValueError(f"expected joints {names}, found {found}")
        self._joint_ids = torch.tensor(local_ids, dtype=torch.long, device=self.device)
        self._range = torch.tensor(
            [env.sim.mj_model.jnt_range[env.sim.mj_model.joint(f"fish/{name}").id, 1]
             for name in names], dtype=torch.float32, device=self.device)
        self._low = torch.tensor([cfg.frequency_range[0], cfg.amplitude_range[0],
                                  cfg.phase_lag_range[0], cfg.steering_range[0]],
                                 device=self.device)
        self._high = torch.tensor([cfg.frequency_range[1], cfg.amplitude_range[1],
                                   cfg.phase_lag_range[1], cfg.steering_range[1]],
                                  device=self.device)
        self.raw = torch.zeros((self.num_envs, 4), device=self.device)
        self.smoothed = torch.zeros_like(self.raw)
        self.previous = torch.zeros_like(self.raw)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.q_target = torch.zeros((self.num_envs, cfg.n_joints), device=self.device)
        self._index = torch.arange(cfg.n_joints, device=self.device, dtype=torch.float32)
        self.reset(slice(None))

    def process_actions(self, actions: torch.Tensor) -> None:
        self.raw[:] = actions.to(self.device)
        normalized = torch.tanh(self.raw)
        mapped = self._low + (normalized + 1.0) * 0.5 * (self._high - self._low)
        self.previous[:] = self.smoothed
        self.smoothed[:] = self.cfg.keep_previous * self.smoothed + (1-self.cfg.keep_previous) * mapped
        self.phase[:] = torch.remainder(
            self.phase + 2*math.pi*self.smoothed[:, 0]*self._env.step_dt, 2*math.pi)
        n = self.cfg.n_joints
        amp = self.smoothed[:, 1:2] * (self.cfg.amplitude_head_ratio +
                (1-self.cfg.amplitude_head_ratio)*self._index[None, :]/(n-1))
        target = self.smoothed[:, 3:4] + amp * torch.sin(
            self.phase[:, None] - 2*math.pi*self.smoothed[:, 2:3]*self._index[None, :]/n)
        if self.cfg.ramp_seconds > 0:
            fade = ((self._env.episode_length_buf.float()+1)*self._env.step_dt /
                    self.cfg.ramp_seconds).clamp(max=1.0)
            target = target * fade[:, None]
        self.q_target[:] = torch.clamp(target, -self._range, self._range)
