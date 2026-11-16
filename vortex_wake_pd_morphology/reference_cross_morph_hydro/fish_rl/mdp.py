"""Goal command, observations, rewards, terminations, and episode metrics."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


@dataclass(kw_only=True)
class FishGoalCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1_000_000.0, 1_000_000.0)
    distance_range: tuple[float, float] = (1.0, 4.0)
    angle_range: tuple[float, float] = (-math.pi/2, math.pi/2)
    marker_name: str = "goal_marker"

    def build(self, env):
        return FishGoal(self, env)


class FishGoal(CommandTerm):
    cfg: FishGoalCfg

    def __init__(self, cfg: FishGoalCfg, env):
        super().__init__(cfg, env)
        self.goal_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)
        body_id = env.sim.mj_model.body(cfg.marker_name).id
        self.marker_id = int(env.sim.mj_model.body_mocapid[body_id])
        assert self.marker_id >= 0

    @property
    def command(self):
        return self.goal_xy

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        radius = self.cfg.distance_range[0] + torch.rand(count, device=self.device) * (
            self.cfg.distance_range[1] - self.cfg.distance_range[0])
        angle = self.cfg.angle_range[0] + torch.rand(count, device=self.device) * (
            self.cfg.angle_range[1] - self.cfg.angle_range[0])
        self.goal_xy[env_ids] = radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
        self.prev_distance[env_ids] = radius
        self._env.sim.data.mocap_pos[env_ids, self.marker_id, :2] = self.goal_xy[env_ids]
        self._env.sim.data.mocap_pos[env_ids, self.marker_id, 2] = 0.5

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass

    def _update_metrics(self) -> None:
        pass


def _fish(env):
    return env.scene["fish"]


def _goal(env) -> FishGoal:
    return env.command_manager.get_term("goal")


def _wave(env):
    return env.action_manager.get_term("wave")


def distance(env) -> torch.Tensor:
    head_xy = _fish(env).data.root_link_pos_w[:, :2]
    return torch.linalg.norm(_goal(env).goal_xy - head_xy, dim=-1)


def target_body(env) -> torch.Tensor:
    fish = _fish(env)
    delta = torch.zeros((env.num_envs, 3), device=env.device)
    delta[:, :2] = _goal(env).goal_xy - fish.data.root_link_pos_w[:, :2]
    return quat_apply_inverse(fish.data.root_link_quat_w, delta)[:, :2]


def obs_target(env) -> torch.Tensor:
    return torch.cat((target_body(env), distance(env)[:, None]), dim=-1)


def obs_linear_velocity(env) -> torch.Tensor:
    return _fish(env).data.root_link_lin_vel_b


def obs_angular_velocity(env) -> torch.Tensor:
    return _fish(env).data.root_link_ang_vel_b


def obs_down_reference(env) -> torch.Tensor:
    # Gravity is zero in the source scene; fixed world -Z is orientation only.
    down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    return quat_apply_inverse(_fish(env).data.root_link_quat_w, down)


def obs_joint_pos(env) -> torch.Tensor:
    return _fish(env).data.joint_pos


def obs_joint_vel(env) -> torch.Tensor:
    return _fish(env).data.joint_vel


def obs_last_action(env) -> torch.Tensor:
    return _wave(env).smoothed


def obs_phase(env) -> torch.Tensor:
    p = _wave(env).phase
    return torch.stack((p.sin(), p.cos()), dim=-1)


def progress(env) -> torch.Tensor:
    current = distance(env)
    result = _goal(env).prev_distance - current
    _goal(env).prev_distance[:] = current
    return result


def heading(env) -> torch.Tensor:
    target = target_body(env)
    return target[:, 0] / torch.linalg.norm(target, dim=-1).clamp_min(1e-6)


def reach(env) -> torch.Tensor:
    return (distance(env) < 0.5).float()


def energy(env) -> torch.Tensor:
    return _fish(env).data.actuator_force.square().sum(dim=-1)


def smoothness(env) -> torch.Tensor:
    wave = _wave(env)
    return (wave.smoothed - wave.previous).square().sum(dim=-1)


def near_limit(env) -> torch.Tensor:
    q = _fish(env).data.joint_pos.abs()
    limits = _wave(env)._range
    return torch.relu(q / limits - 0.9).sum(dim=-1)


def reached(env) -> torch.Tensor:
    return distance(env) < 0.5


def timed_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def diverged(env, speed_limit: float = 20.0) -> torch.Tensor:
    qpos = env.sim.data.qpos
    qvel = env.sim.data.qvel
    finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
    return ~finite | (torch.linalg.norm(qvel[:, :3], dim=-1) > speed_limit)


def out_of_bounds(env, radius: float = 8.0) -> torch.Tensor:
    return torch.linalg.norm(_fish(env).data.root_link_pos_w[:, :2], dim=-1) > radius


def metric_success(env) -> torch.Tensor:
    return reached(env).float()


def metric_length(env) -> torch.Tensor:
    return env.episode_length_buf.float() * env.step_dt


def metric_distance(env) -> torch.Tensor:
    return distance(env)
